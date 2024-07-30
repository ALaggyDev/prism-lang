use gc::Gc;
use prism_lang::bytecode::Vm;
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

    let code_object = Gc::new(compile(&program, &interner).unwrap());
    println!("{:?}", code_object);

    let mut vm = Vm::new_from_code_object(Gc::clone(&code_object), &[], true);

    while vm.result.is_none() {
        vm.step();
    }

    println!("{:?}", vm.result);

    Ok(())
}
