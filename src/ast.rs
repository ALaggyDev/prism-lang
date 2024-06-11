use std::rc::Rc;

use crate::token::{Ident, Literal, Token};

#[derive(Clone, Debug)]
pub struct Parser<'cx> {
    rem: &'cx [Token<'cx>],
}

#[derive(Clone, Debug)]
pub struct CompileError(pub Box<str>);

impl<'cx> Parser<'cx> {
    pub fn new(tokens: &'cx [Token<'cx>]) -> Self {
        Self { rem: tokens}
    }

    pub fn advance(&mut self) {
        if !self.rem.is_empty() {
            self.rem = &self.rem[1..]
        }
    }

    pub fn peek(&self) -> &Token<'cx> {
        if !self.rem.is_empty() {
            &self.rem[0]
        } else {
            &Token::Eof
        }
    }

    pub fn read(&mut self) -> Token<'cx> {
        let token = self.peek().clone();
        self.advance();
        token
    }

    pub fn is_at_end(&self) -> bool {
        self.rem.is_empty()
    }
}

macro_rules! expect_token {
    ($parser: expr, $pat: pat, $msg: literal) => {
        let $pat = $parser.read() else {
            return Err(CompileError($msg.into()));
        };
    };
}

macro_rules! read_vec {
    ($parser: expr, $item_fn: expr, $is_sep: pat, $is_delim: pat) => {{
        let mut items = Vec::new();

        if let $is_delim = $parser.peek() {
            $parser.advance();
        } else {
            loop {
                let item = $item_fn($parser)?;
                items.push(item);

                match $parser.read() {
                    $is_sep => {}
                    $is_delim => break,
                    _ => return Err(CompileError("Unexpected token.".into())),
                }
            }
        }

        items
    }};
}

pub trait Parse<'cx>: Sized {
    fn parse(parser: &mut Parser<'cx>) -> Result<Self, CompileError>;
}

#[derive(Clone, Debug)]
pub enum Expr<'cx> {
    Literal(Literal),
    Variable(Ident<'cx>),
    FunctionCall(Box<Expr<'cx>>, Box<[Expr<'cx>]>),
    Grouped(Box<Expr<'cx>>),
    Unary(UnaryOp, Box<Expr<'cx>>),
    Binary(BinaryOp, Box<Expr<'cx>>, Box<Expr<'cx>>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Assign,

    Plus,
    Minus,
    Multiply,
    Divide,

    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
    And,
    Or,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fixity {
    // Left-associative
    Left,
    // Reft-associative
    Right,
}

impl<'cx> Parse<'cx> for Ident<'cx> {
    fn parse(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        match parser.read() {
            Token::Ident(ident) => Ok(ident),
            _ => Err(CompileError("Expected ident.".into())),
        }
    }
}

impl BinaryOp {
    pub fn new(token: &Token) -> Option<BinaryOp> {
        match token {
            Token::Assign => Some(Self::Assign),
            Token::Plus => Some(Self::Plus),
            Token::Minus => Some(Self::Minus),
            Token::Multiply => Some(Self::Multiply),
            Token::Divide => Some(Self::Divide),
            Token::Equal => Some(Self::Equal),
            Token::NotEqual => Some(Self::NotEqual),
            Token::Greater => Some(Self::Greater),
            Token::Less => Some(Self::Less),
            Token::GreaterOrEqual => Some(Self::GreaterOrEqual),
            Token::LessOrEqual => Some(Self::LessOrEqual),
            Token::And => Some(Self::And),
            Token::Or => Some(Self::Or),
            _ => None,
        }
    }

    pub fn precedence(self) -> usize {
        match self {
            Self::Multiply | Self::Divide => 4,
            Self::Plus | Self::Minus => 3,
            Self::Equal
            | Self::NotEqual
            | Self::Greater
            | Self::Less
            | Self::GreaterOrEqual
            | Self::LessOrEqual => 2,
            Self::And | Self::Or => 1,
            Self::Assign => 0,
        }
    }

    pub fn fixity(self) -> Fixity {
        match self {
            Self::Plus
            | Self::Minus
            | Self::Multiply
            | Self::Divide
            | Self::Equal
            | Self::NotEqual
            | Self::Greater
            | Self::Less
            | Self::GreaterOrEqual
            | Self::LessOrEqual
            | Self::And
            | Self::Or => Fixity::Left,
            Self::Assign => Fixity::Right,
        }
    }
}

impl<'cx> Expr<'cx> {
    fn parse_primary(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        let token = parser.read();

        Ok(match token {
            Token::Literal(literal) => Expr::Literal(literal),
            Token::OpenParen => {
                let expr = Expr::parse(parser)?;
                expect_token!(parser, Token::CloseParen, "Expected closing paraenesis.");
                Expr::Grouped(expr.into())
            }
            Token::Minus => {
                Expr::Unary(UnaryOp::Negate, Box::new(Expr::parse_primary_func(parser)?))
            }
            Token::Not => Expr::Unary(UnaryOp::Not, Box::new(Expr::parse_primary_func(parser)?)),
            Token::Ident(ident) => Expr::Variable(ident),
            _ => return Err(CompileError("Unexpected token.".into())),
        })
    }

    fn parse_primary_func(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        let mut lhs = Expr::parse_primary(parser)?;

        while let Token::OpenParen = parser.peek() {
            parser.advance();

            let exprs = read_vec!(parser, Expr::parse, Token::Comma, Token::CloseParen);

            lhs = Expr::FunctionCall(lhs.into(), exprs.into());
        }

        Ok(lhs)
    }

    fn parse_min_pred(
        parser: &mut Parser<'cx>,
        mut lhs: Expr<'cx>,
        min_pred: usize,
    ) -> Result<Self, CompileError> {
        // See: https://en.wikipedia.org/wiki/Operator-precedence_parser

        // parse_expression()
        //     return parse_expression_1(parse_primary(), 0)
        // parse_expression_1(lhs, min_precedence)
        //     lookahead := peek next token
        //     while lookahead is a binary operator whose precedence is >= min_precedence
        //         op := lookahead
        //         advance to next token
        //         rhs := parse_primary ()
        //         lookahead := peek next token
        //         while lookahead is a binary operator whose precedence is greater
        //                  than op's, or a right-associative operator
        //                  whose precedence is equal to op's
        //             rhs := parse_expression_1 (rhs, precedence of op + (1 if lookahead precedence is greater, else 0))
        //             lookahead := peek next token
        //         lhs := the result of applying op with operands lhs and rhs
        //     return lhs

        let mut lookahead = parser.peek().clone();

        while let Some(op1) = BinaryOp::new(&lookahead) {
            if op1.precedence() < min_pred {
                break;
            }

            parser.advance();

            let mut rhs = Expr::parse_primary_func(parser)?;

            lookahead = parser.peek().clone();

            while let Some(op2) = BinaryOp::new(&lookahead) {
                if !(op2.precedence() > op1.precedence()
                    || (op2.fixity() == Fixity::Right && op2.precedence() == op1.precedence()))
                {
                    break;
                }

                rhs = Expr::parse_min_pred(
                    parser,
                    rhs,
                    op1.precedence() + (op2.precedence() > op1.precedence()) as usize,
                )?;
                lookahead = parser.peek().clone();
            }

            lhs = Expr::Binary(op1, Box::new(lhs), Box::new(rhs));
        }

        Ok(lhs)
    }
}

impl<'cx> Parse<'cx> for Expr<'cx> {
    fn parse(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        let primary = Expr::parse_primary_func(parser)?;

        Self::parse_min_pred(parser, primary, 0)
    }
}

#[derive(Clone, Debug)]
pub enum Stmt<'cx> {
    Expr(Expr<'cx>),
    Block(Block<'cx>),
    Fn(Ident<'cx>, Rc<FunctionDecl<'cx>>),
    Let(Ident<'cx>, Expr<'cx>),
    If(Box<[(Expr<'cx>, Block<'cx>)]>, Option<Block<'cx>>),
    While(Expr<'cx>, Block<'cx>),
    Break,
    Continue,
    Return(Expr<'cx>),
}

#[derive(Clone, Debug)]
pub struct FunctionDecl<'cx> {
    pub parameters: Box<[Ident<'cx>]>,
    pub block: Block<'cx>,
}

impl<'cx> Parse<'cx> for Stmt<'cx> {
    fn parse(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        match parser.peek() {
            Token::OpenBrace => Ok(Self::Block(Block::parse(parser)?)),
            Token::Fn => {
                parser.advance();

                let ident = Ident::parse(parser)?;

                expect_token!(parser, Token::OpenParen, "Expected opening paraenesis.");

                let parameters = read_vec!(parser, Ident::parse, Token::Comma, Token::CloseParen);

                let block = Block::parse(parser)?;

                Ok(Self::Fn(
                    ident,
                    Rc::new(FunctionDecl {
                        parameters: parameters.into(),
                        block,
                    }),
                ))
            }
            Token::Let => {
                parser.advance();

                let ident = Ident::parse(parser)?;
                expect_token!(parser, Token::Assign, "Expected assignment.");

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon, "Expected semicolon.");

                Ok(Self::Let(ident, expr))
            }
            Token::If => {
                parser.advance();

                let mut branches = vec![];
                let mut else_branch = None;

                loop {
                    expect_token!(parser, Token::OpenParen, "Expected opening paraenesis.");

                    let expr = Expr::parse(parser)?;

                    expect_token!(parser, Token::CloseParen, "Expected closing paraenesis.");

                    let block = Block::parse(parser)?;

                    branches.push((expr, block));

                    if let Token::Else = parser.peek() {
                        parser.advance();

                        match parser.peek() {
                            Token::If => {
                                parser.advance();
                            }
                            Token::OpenBrace => {
                                else_branch = Some(Block::parse(parser)?);
                                break;
                            }
                            _ => return Err(CompileError("Expected if or opening brace.".into())),
                        }
                    } else {
                        break;
                    }
                }

                Ok(Self::If(branches.into(), else_branch))
            }
            Token::While => {
                parser.advance();

                expect_token!(parser, Token::OpenParen, "Expected opening paraenesis.");

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::CloseParen, "Expected closing paraenesis.");

                let block = Block::parse(parser)?;

                Ok(Self::While(expr, block))
            }
            Token::Break => {
                parser.advance();

                expect_token!(parser, Token::Semicolon, "Expected semicolon.");

                Ok(Self::Break)
            }
            Token::Continue => {
                parser.advance();

                expect_token!(parser, Token::Semicolon, "Expected semicolon.");

                Ok(Self::Continue)
            }
            Token::Return => {
                parser.advance();

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon, "Expected semicolon.");

                Ok(Self::Return(expr))
            }
            _ => {
                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon, "Expected semicolon.");

                Ok(Self::Expr(expr))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block<'cx>(pub Box<[Stmt<'cx>]>);

impl<'cx> Parse<'cx> for Block<'cx> {
    fn parse(parser: &mut Parser<'cx>) -> Result<Self, CompileError> {
        expect_token!(parser, Token::OpenBrace, "Expected opening brace.");

        let mut stmts = vec![];

        loop {
            if let Token::CloseBrace = parser.peek() {
                parser.advance();
                break;
            };

            stmts.push(Stmt::parse(parser)?);
        }

        Ok(Self(stmts.into()))
    }
}
