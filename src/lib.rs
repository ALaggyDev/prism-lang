use ast::{CompileError, Parse, Parser, Stmt};
use logos::Logos;
use string_interner::{DefaultBackend, StringInterner};
use token::Token;

pub mod ast;
pub mod bytecode;
pub mod call_frame;
pub mod compiler;
pub mod native_func;
pub mod token;
pub mod vm;

pub fn lex<'a>(
    program: &'a str,
    interner: &'a mut StringInterner<DefaultBackend>,
) -> Result<Vec<Token>, &'a str> {
    let mut lex = Token::lexer_with_extras(program, interner);

    let mut tokens = Vec::new();

    while let Some(token) = lex.next() {
        if let Ok(token) = token {
            tokens.push(token);
        } else {
            return Err(lex.slice());
        }
    }

    Ok(tokens)
}

pub fn parse(tokens: &[Token], interactive: bool) -> Result<Vec<Stmt>, CompileError> {
    let mut parser = Parser::new(tokens, interactive);

    let mut stmts = vec![];

    while !parser.is_at_end() {
        stmts.push(Stmt::parse(&mut parser)?);
    }

    Ok(stmts)
}
