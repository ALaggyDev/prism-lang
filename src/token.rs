use std::rc::Rc;

use logos::Logos;

#[derive(Logos, Clone, Debug, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"\/\/[^\r\n]*")] // Line comment
#[logos(skip r"\/\*([^*]|\*[^/])*\*\/")] // Block comment (unnested)
pub enum Token {
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| Ident(lex.slice().into()), priority = 1)]
    Ident(Ident),

    #[token("null", |_| Literal::Null)]
    #[token("true", |_| Literal::Bool(true))]
    #[token("false", |_| Literal::Bool(false))]
    #[token("Infinity", |_| Literal::Number(f64::INFINITY))]
    #[token("NaN", |_| Literal::Number(f64::NAN))]
    #[regex(r"([0-9]*\.)?[0-9]+([eE][+-]?[0-9]+)?", |lex| Literal::Number(lex.slice().parse::<f64>().unwrap()))]
    #[regex(r#""[^"\r\n]*""#, |lex| Literal::String(lex.slice()[1..lex.slice().len() - 1].into()))]
    Literal(Literal),

    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,

    #[token("=")]
    Assign,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,

    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token("{")]
    OpenBrace,
    #[token("}")]
    CloseBrace,
    #[token("[")]
    OpenBracket,
    #[token("]")]
    CloseBracket,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Multiply,
    #[token("/")]
    Divide,
    #[token("==")]
    Equal,
    #[token("!=")]
    NotEqual,
    #[token(">")]
    Greater,
    #[token("<")]
    Less,
    #[token(">=")]
    GreaterOrEqual,
    #[token("<=")]
    LessOrEqual,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,

    Eof,
}

// TODO: Make Ident not owned and Copy, e.g. multiple Ident can point to the same memory
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ident(pub Box<str>);

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Number(f64),
    String(Rc<str>),
}
