use std::io::{self, Write};

use gc_arena::{Gc, Mutation};
use itertools::Itertools;

use crate::vm::{Callable, NativeFunc, Value};

pub fn print<'gc>(args: &[Value<'gc>], _: &Mutation<'gc>) -> Value<'gc> {
    // If objects have circular references, a naive print approach will crash the program.
    // Therefore we need to store the objects we have visited.

    // Using *const () is pretty cursed, but we have no better way since the type of the gc-managed object is unknown.
    let mut visited: Vec<*const ()> = vec![];

    let output = args
        .iter()
        .map(|arg| print_inner(&mut visited, arg))
        .join(" ");

    println!("{}", output);
    Value::Null
}

fn print_inner(visited: &mut Vec<*const ()>, value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(val) => format!("{}", val),
        Value::Number(val) => format!("{}", val),
        Value::String(val) => format!("{}", val),
        Value::Tuple(tuple) => {
            let ptr = tuple.as_ref() as *const _ as *const ();

            if visited.contains(&ptr) {
                "(...)".into()
            } else {
                visited.push(ptr);

                let output = tuple.iter().map(|e| print_inner(visited, e)).join(", ");

                visited.pop();

                format!("({})", output)
            }
        }
        Value::Array(array) => {
            let ptr = array.as_ref() as *const _ as *const ();

            if visited.contains(&ptr) {
                "[...]".into()
            } else {
                visited.push(ptr);

                let output = array
                    .borrow()
                    .iter()
                    .map(|e| print_inner(visited, e))
                    .join(", ");

                visited.pop();

                format!("[{}]", output)
            }
        }
        Value::Callable(Callable::Func(_)) => "<func>".into(),
        Value::Callable(Callable::Native(_)) => "<native func>".into(),
    }
}

pub fn input<'gc>(args: &[Value<'gc>], mc: &Mutation<'gc>) -> Value<'gc> {
    if !args.is_empty() {
        let mut visited: Vec<*const ()> = vec![];
        print!("{}", print_inner(&mut visited, &args[0]));
        io::stdout().flush().unwrap();
    }

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    trim_newline(&mut input);

    Value::String(Gc::new(mc, input.into()))
}

fn trim_newline(s: &mut String) {
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
}

pub static NATIVE_FUNCS: &[(&str, NativeFunc)] =
    &[("print", NativeFunc(print)), ("input", NativeFunc(input))];
