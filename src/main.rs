use ast::{Parse, Parser, Stmt};
use interpreter::{ControlFlow, Evalulate, Function, Interpreter, Value};
use logos::Logos;
use native_func::NATIVE_FUNCS;
use token::Token;

pub mod ast;
pub mod interpreter;
pub mod native_func;
pub mod token;

fn stage_1(program: &str) -> Vec<Token> {
    let mut lex = Token::lexer(program);

    let mut tokens = Vec::new();
    let mut errored = false;

    while let Some(token) = lex.next() {
        if let Ok(token) = token {
            tokens.push(token);
        } else {
            errored = true;
            println!("Unknown token at {}..{}.", lex.span().start, lex.span().end);
        }
    }

    if errored {
        panic!("Exit");
    }

    tokens
}

fn stage_2(tokens: &[Token]) -> Vec<Stmt> {
    let mut parser = Parser::new(tokens);

    let mut stmts = vec![];

    while !parser.is_at_end() {
        stmts.push(Stmt::parse(&mut parser));
    }

    stmts
}

fn interpret(program: &[Stmt]) -> Result<(), ControlFlow> {
    let mut ins = Interpreter::new();

    for (ident, func) in NATIVE_FUNCS.iter() {
        ins.add_var(ident.clone(), Value::Function(Function::Native(*func)));
    }

    for stmt in program {
        stmt.evalulate(&mut ins)?;
    }

    Ok(())
}

fn main() {
    println!("Prism!");

    let tokens = stage_1(include_str!("./test.prism"));
    let program = stage_2(&tokens);

    let res = interpret(&program);
    println!("{:?}", res);
}
