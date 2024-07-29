#![allow(non_local_definitions)] // for gc-derive in miri run

use ast::{CompileError, Parse, Parser, Stmt};
use logos::Logos;
use string_interner::{DefaultBackend, StringInterner};
use token::Token;

pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod native_func;
pub mod token;

pub fn stage_1(program: &str) -> (Vec<Token>, StringInterner<DefaultBackend>) {
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

pub fn stage_2(tokens: &[Token]) -> Result<Vec<Stmt>, CompileError> {
    let mut parser = Parser::new(tokens);

    let mut stmts = vec![];

    while !parser.is_at_end() {
        stmts.push(Stmt::parse(&mut parser)?);
    }

    Ok(stmts)
}
