use crate::{
    ast::{BinaryOp, Block, Expr, Stmt, UnaryOp},
    token::Literal,
};

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(Box<str>),
}

#[derive(Clone, Debug)]
pub struct RuntimeError(pub Box<str>);

pub trait Evalulate: Sized {
    fn evalulate(&self) -> Result<Value, RuntimeError>;
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(val) => *val,
            Value::Number(val) => *val != 0.0 && !(*val).is_nan(),
            Value::String(val) => !(*val).is_empty(),
        }
    }

    pub fn is_equal(&self, rhs: &Self) -> bool {
        // Note: Make sure to update this table when a new value is added
        match (self, rhs) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(val_1), Value::Bool(val_2)) => val_1 == val_2,
            (Value::Number(val_1), Value::Number(val_2)) => val_1 == val_2,
            (Value::String(val_1), Value::String(val_2)) => val_1 == val_2,
            _ => false,
        }
    }
}

impl UnaryOp {
    fn evalulate_op(&self, op1: &Expr) -> Result<Value, RuntimeError> {
        let op1 = op1.evalulate()?;

        match self {
            UnaryOp::Negation => match op1 {
                Value::Number(val) => Ok(Value::Number(-val)),
                _ => Err(RuntimeError("Negation requires number.".into())),
            },
            UnaryOp::Not => Ok(Value::Bool(!op1.is_truthy())),
        }
    }
}

impl BinaryOp {
    fn evalulate_op(&self, op1: &Box<Expr>, op2: &Expr) -> Result<Value, RuntimeError> {
        // Assignment have special treatment
        if let BinaryOp::Assign = self {
            todo!()
            // TODO: return
        }

        let op1 = op1.evalulate()?;
        let op2 = op2.evalulate()?;

        Ok(match self {
            BinaryOp::Assign => unreachable!(),
            BinaryOp::Plus
            | BinaryOp::Minus
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Greater
            | BinaryOp::Less
            | BinaryOp::GreaterOrEqual
            | BinaryOp::LessOrEqual => {
                let Value::Number(val_1) = op1 else {
                    return Err(RuntimeError("Expected number.".into()));
                };
                let Value::Number(val_2) = op2 else {
                    return Err(RuntimeError("Expected number.".into()));
                };

                match self {
                    BinaryOp::Plus => Value::Number(val_1 + val_2),
                    BinaryOp::Minus => Value::Number(val_1 - val_2),
                    BinaryOp::Multiply => Value::Number(val_1 * val_2),
                    BinaryOp::Divide => Value::Number(val_1 / val_2),
                    BinaryOp::Greater => Value::Bool(val_1 > val_2),
                    BinaryOp::Less => Value::Bool(val_1 < val_2),
                    BinaryOp::GreaterOrEqual => Value::Bool(val_1 >= val_2),
                    BinaryOp::LessOrEqual => Value::Bool(val_1 <= val_2),
                    _ => unreachable!(),
                }
            }

            BinaryOp::Equal => Value::Bool(op1.is_equal(&op2)),
            BinaryOp::NotEqual => Value::Bool(!op1.is_equal(&op2)),

            BinaryOp::And => Value::Bool(op1.is_truthy() && op2.is_truthy()),
            BinaryOp::Or => Value::Bool(op1.is_truthy() || op2.is_truthy()),
        })
    }
}

impl Evalulate for Expr {
    fn evalulate(&self) -> Result<Value, RuntimeError> {
        Ok(match self {
            Expr::Literal(val) => match val {
                Literal::Null => Value::Null,
                Literal::Bool(val) => Value::Bool(*val),
                Literal::Number(val) => Value::Number(*val),
                Literal::String(val) => Value::String(val.clone()), // TODO: Don't use clone
            },
            Expr::Variable(_) => todo!(),
            Expr::FunctionCall(_, _) => todo!(),
            Expr::Unary(op, op_1) => op.evalulate_op(op_1)?,
            Expr::Binary(op, op_1, op_2) => op.evalulate_op(op_1, op_2)?,
        })
    }
}

impl Evalulate for Stmt {
    fn evalulate(&self) -> Result<Value, RuntimeError> {
        match self {
            Stmt::Expr(expr) => println!("Stmt expr eval result: {:?}", expr.evalulate()?),
            Stmt::Fn(_, _, _) => todo!(),
            Stmt::Return(_) => todo!(),
            Stmt::Let(_, _) => todo!(),
            Stmt::If(branches, else_branch) => 'outer: {
                for (cond, branch) in branches.iter() {
                    if cond.evalulate()?.is_truthy() {
                        branch.evalulate()?;
                        break 'outer;
                    }
                }

                if let Some(else_branch) = else_branch {
                    else_branch.evalulate()?;
                }
            }
        };

        Ok(Value::Null)
    }
}

impl Evalulate for Block {
    fn evalulate(&self) -> Result<Value, RuntimeError> {
        for stmt in self.0.iter() {
            stmt.evalulate()?;
        }

        Ok(Value::Null)
    }
}
