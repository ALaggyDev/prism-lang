use logos::Logos;
use prism_lang::ast::{CompileError, Parse, Parser, Stmt};
use prism_lang::token::Token;
use std::{env, fs, io};
use string_interner::{DefaultBackend, StringInterner};

fn stage_1(program: &str) -> (Vec<Token>, StringInterner<DefaultBackend>) {
    let mut lex = Token::lexer_with_extras(program, StringInterner::new());

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

    (tokens, lex.extras)
}

fn stage_2(tokens: &[Token]) -> Result<Vec<Stmt>, CompileError> {
    let mut parser = Parser::new(tokens);

    let mut stmts = vec![];

    while !parser.is_at_end() {
        stmts.push(Stmt::parse(&mut parser)?);
    }

    Ok(stmts)
}

fn main() -> Result<(), io::Error> {
    println!("Prism!");

    let pathname = env::args()
        .nth(1)
        .expect("Please provide a path to the script.");
    let content = fs::read_to_string(pathname)?;

    let (tokens, _) = stage_1(&content);
    let program = stage_2(&tokens).unwrap();

    println!("{:?}", program);

    Ok(())
}
