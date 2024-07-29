use gc::Gc;
use prism_lang::bytecode::{Callable, Value, Vm};
use prism_lang::compiler::compile;
use prism_lang::{stage_1, stage_2};
use std::{env, fs, io};

fn main() -> Result<(), io::Error> {
    println!("Prism!");

    let pathname = env::args()
        .nth(1)
        .expect("Please provide a path to the script.");
    let content = fs::read_to_string(pathname)?;

    let (tokens, interner) = stage_1(&content);
    let program = stage_2(&tokens).unwrap();

    println!("{:?}", program);

    let code_object = Gc::new(compile(&program, &interner).unwrap());
    println!("{:?}", code_object);

    let mut vm = Vm::new_from_code_object(Gc::clone(&code_object), &[Value::Number(30.0)], true);

    vm.globals
        .insert("fib".into(), Value::Callable(Callable::Func(code_object)));

    while vm.result.is_none() {
        vm.step();
    }

    println!("{:?}", vm.result);

    Ok(())
}
