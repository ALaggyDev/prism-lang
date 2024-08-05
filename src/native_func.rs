use itertools::Itertools;

use crate::bytecode::{Callable, NativeFunc, Value};

// If objects have circular references, a naive print approach will crash the program.
// Therefore we need to store the objects we have visited.
pub fn print(args: &[Value]) -> Value {
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

pub static NATIVE_FUNCS: &[(&str, NativeFunc)] = &[("print", print)];
