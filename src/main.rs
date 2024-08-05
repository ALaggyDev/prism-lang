use gc::Gc;
use prism_lang::ast::{Expr, Stmt};
use prism_lang::bytecode::Vm;
use prism_lang::compiler::compile;
use prism_lang::token::{Ident, Token};
use prism_lang::{lex, parse};
use std::io::Write;
use std::{env, fs, io};
use string_interner::StringInterner;

fn interactive_mode() -> io::Result<()> {
    println!("Prism language 1.0.0. Interactive shell.\nType \".help\" for more information.");

    let mut interner = StringInterner::new();
    let mut tokens: Vec<Token> = vec![];

    let mut more_tokens = false;

    let mut vm = Vm::new(true);

    loop {
        // Print ... or >>>
        if more_tokens {
            print!("... ");
        } else {
            print!(">>> ");
        }
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        // Print help message
        if input.starts_with(".help") {
            println!("There is no help message. Go figure out the solution yourself.");

            continue;
        }

        tokens.append(&mut lex(&input, &mut interner));

        match parse(&tokens, true) {
            Ok(mut program) => {
                tokens.clear();

                more_tokens = false;

                // Print the result if only one expression exists
                // Another stupid hack to make this work
                if let [Stmt::Expr(expr)] = &mut program[..] {
                    *expr = Expr::FunctionCall(
                        Box::new(Expr::Variable(Ident(
                            interner.get_or_intern_static("print"),
                        ))),
                        Box::new([expr.clone()]),
                    );
                }

                // Compile the program
                let code_object = Gc::new(compile(&program, &interner).unwrap());

                // Execute
                vm.push_frame(code_object, &[]);

                vm.finished = false;
                while !vm.finished {
                    vm.step();
                }
            }

            // the input simply is not finished, we keep waiting for more input
            Err(err) if *err.0 == Token::Eof => {
                more_tokens = true;
            }

            Err(err) => {
                tokens.clear();
                more_tokens = false;

                println!("Unexpected token: {:?}", err.0);
            }
        }
    }
}

fn main() -> Result<(), io::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        return interactive_mode();
    }

    let pathname = args.get(1).expect("Please provide a path to the script.");

    let content = fs::read_to_string(pathname)?;

    let mut interner = StringInterner::new();
    let tokens = lex(&content, &mut interner);
    let program = parse(&tokens, false).unwrap();

    let code_object = Gc::new(compile(&program, &interner).unwrap());

    let mut vm = Vm::new(true);
    vm.push_frame(code_object, &[]);

    while !vm.finished {
        vm.step();
    }

    Ok(())
}
