use crate::token::{Ident, Literal, Token};

#[derive(Clone, Debug)]
pub struct Parser<'a> {
    rem: &'a [Token],
    interactive: bool,
}

#[derive(Debug)]
pub struct CompileError(pub Box<Token>);

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token], interactive: bool) -> Self {
        Self {
            rem: tokens,
            interactive,
        }
    }

    pub fn advance(&mut self) {
        if !self.rem.is_empty() {
            self.rem = &self.rem[1..]
        }
    }

    pub fn peek(&self) -> &Token {
        if !self.rem.is_empty() {
            &self.rem[0]
        } else {
            &Token::Eof
        }
    }

    pub fn read(&mut self) -> Token {
        let token = self.peek().clone();
        self.advance();
        token
    }

    pub fn is_at_end(&self) -> bool {
        self.rem.is_empty()
    }
}

macro_rules! compile_err {
    ($token: expr) => {
        return Err(CompileError(Box::new($token)))
    };
}

macro_rules! expect_token {
    // Stupid hack to make interactive shell more usable
    ($parser: expr, Token::Semicolon) => {
        let token = $parser.read();
        if token != Token::Semicolon && !($parser.interactive && token == Token::Eof) {
            compile_err!(token);
        }
    };
    ($parser: expr, $pat: pat) => {
        let token = $parser.read();
        let $pat = token else {
            compile_err!(token);
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
                    other => compile_err!(other),
                }
            }
        }

        items
    }};
}

pub trait Parse: Sized {
    fn parse(parser: &mut Parser) -> Result<Self, CompileError>;
}

#[derive(Clone, Debug)]
pub enum Expr {
    Literal(Literal),
    Variable(Ident),
    FunctionCall(Box<Expr>, Box<[Expr]>),
    Grouped(Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Tuple(Box<[Expr]>),
    Array(Box<[Expr]>),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    Assign,

    Add,
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

impl Parse for Ident {
    fn parse(parser: &mut Parser) -> Result<Self, CompileError> {
        expect_token!(parser, Token::Ident(ident));

        Ok(ident)
    }
}

impl BinaryOp {
    pub fn new(token: &Token) -> Option<BinaryOp> {
        match token {
            Token::Assign => Some(Self::Assign),
            Token::Plus => Some(Self::Add),
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
            Self::Add | Self::Minus => 3,
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
            Self::Add
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

impl Expr {
    fn parse_primary(parser: &mut Parser) -> Result<Self, CompileError> {
        // Left operators
        let token = parser.read();

        let mut lhs = match token {
            Token::Literal(literal) => Expr::Literal(literal),
            // Can be grouped or tuple
            Token::OpenParen => {
                let expr = Expr::parse(parser)?;

                match parser.read() {
                    Token::CloseParen => Expr::Grouped(expr.into()),
                    Token::Comma => {
                        let mut exprs = vec![expr];

                        loop {
                            if let Token::CloseParen = parser.peek() {
                                parser.advance();
                                break;
                            }

                            exprs.push(Expr::parse(parser)?);

                            if let Token::Comma = parser.peek() {
                                parser.advance();
                            }
                        }

                        Expr::Tuple(exprs.into())
                    }
                    other => compile_err!(other),
                }
            }
            Token::OpenBracket => {
                let exprs = read_vec!(parser, Expr::parse, Token::Comma, Token::CloseBracket);

                Expr::Array(exprs.into())
            }
            Token::Minus => Expr::Unary(UnaryOp::Negate, Box::new(Expr::parse_primary(parser)?)),
            Token::Not => Expr::Unary(UnaryOp::Not, Box::new(Expr::parse_primary(parser)?)),
            Token::Ident(ident) => Expr::Variable(ident),
            other => compile_err!(other),
        };

        // Right operators

        loop {
            match parser.peek() {
                Token::OpenParen => {
                    parser.advance();

                    let exprs = read_vec!(parser, Expr::parse, Token::Comma, Token::CloseParen);

                    lhs = Expr::FunctionCall(lhs.into(), exprs.into());
                }
                Token::OpenBracket => {
                    parser.advance();

                    let expr = Expr::parse(parser)?;

                    expect_token!(parser, Token::CloseBracket);

                    lhs = Expr::Index(lhs.into(), expr.into());
                }
                _ => break,
            }
        }

        Ok(lhs)
    }

    fn parse_min_pred(
        parser: &mut Parser,
        mut lhs: Expr,
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

            let mut rhs = Expr::parse_primary(parser)?;

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

impl Parse for Expr {
    fn parse(parser: &mut Parser) -> Result<Self, CompileError> {
        let primary = Expr::parse_primary(parser)?;

        Self::parse_min_pred(parser, primary, 0)
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Empty,
    Expr(Expr),
    Block(Block),
    Fn(Ident, FuncDecl),
    Let(Ident, Expr),
    If(Box<[(Expr, Block)]>, Option<Block>),
    While(Expr, Block),
    For(Ident, (Expr, Expr), Block),
    Break,
    Continue,
    Return(Expr),
    Class(ClassDecl),
}

#[derive(Clone, Debug)]
pub struct FuncDecl {
    pub parameters: Box<[Ident]>,
    pub block: Block,
}

#[derive(Clone, Debug)]
pub struct ClassDecl {
    pub ident: Ident,
    pub methods: Box<[(Ident, FuncDecl)]>,
}

impl Stmt {
    fn parse_fn(parser: &mut Parser) -> Result<(Ident, FuncDecl), CompileError> {
        expect_token!(parser, Token::Fn);

        let ident = Ident::parse(parser)?;

        expect_token!(parser, Token::OpenParen);

        let parameters = read_vec!(parser, Ident::parse, Token::Comma, Token::CloseParen);

        let block = Block::parse(parser)?;

        Ok((
            ident,
            FuncDecl {
                parameters: parameters.into(),
                block,
            },
        ))
    }
}

impl Parse for Stmt {
    fn parse(parser: &mut Parser) -> Result<Self, CompileError> {
        match parser.peek() {
            Token::Semicolon => {
                parser.advance();
                Ok(Self::Empty)
            }
            Token::OpenBrace => Ok(Self::Block(Block::parse(parser)?)),
            Token::Fn => {
                let decl = Stmt::parse_fn(parser)?;
                Ok(Self::Fn(decl.0, decl.1))
            }
            Token::Let => {
                parser.advance();

                let ident = Ident::parse(parser)?;
                expect_token!(parser, Token::Assign);

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon);

                Ok(Self::Let(ident, expr))
            }
            Token::If => {
                parser.advance();

                let mut branches = vec![];
                let mut else_branch = None;

                loop {
                    expect_token!(parser, Token::OpenParen);

                    let expr = Expr::parse(parser)?;

                    expect_token!(parser, Token::CloseParen);

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
                            _ => compile_err!(parser.read()),
                        }
                    } else {
                        break;
                    }
                }

                Ok(Self::If(branches.into(), else_branch))
            }
            Token::While => {
                parser.advance();

                expect_token!(parser, Token::OpenParen);

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::CloseParen);

                let block = Block::parse(parser)?;

                Ok(Self::While(expr, block))
            }
            Token::For => {
                parser.advance();

                let ident = Ident::parse(parser)?;

                expect_token!(parser, Token::In);

                let l_expr = Expr::parse(parser)?;
                expect_token!(parser, Token::Range);
                let r_expr = Expr::parse(parser)?;

                let block = Block::parse(parser)?;

                Ok(Self::For(ident, (l_expr, r_expr), block))
            }
            Token::Break => {
                parser.advance();

                expect_token!(parser, Token::Semicolon);

                Ok(Self::Break)
            }
            Token::Continue => {
                parser.advance();

                expect_token!(parser, Token::Semicolon);

                Ok(Self::Continue)
            }
            Token::Return => {
                parser.advance();

                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon);

                Ok(Self::Return(expr))
            }
            Token::Class => {
                parser.advance();

                let ident = Ident::parse(parser)?;

                expect_token!(parser, Token::OpenBrace);

                let mut methods = Vec::new();

                loop {
                    if let Token::CloseBrace = parser.peek() {
                        parser.advance();
                        break;
                    }

                    let res = Stmt::parse_fn(parser)?;
                    methods.push((res.0, res.1));
                }

                Ok(Self::Class(ClassDecl {
                    ident,
                    methods: methods.into(),
                }))
            }
            _ => {
                let expr = Expr::parse(parser)?;

                expect_token!(parser, Token::Semicolon);

                Ok(Self::Expr(expr))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block(pub Box<[Stmt]>);

impl Parse for Block {
    fn parse(parser: &mut Parser) -> Result<Self, CompileError> {
        expect_token!(parser, Token::OpenBrace);

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
