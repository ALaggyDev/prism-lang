use std::collections::HashMap;

use crate::{
    ast::{BinaryOp, Block, Expr, Stmt, UnaryOp},
    token::{Ident, Literal},
};

#[derive(Clone, Debug)]
pub struct Interpreter {
    scopes: Vec<Scope>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    pub fn eval_in_scope(
        &mut self,
        func: impl FnOnce(&mut Self) -> Result<Value, RuntimeError>,
    ) -> Result<Value, RuntimeError> {
        self.scopes.push(Scope::new());

        let res = func(self);

        self.scopes.pop();

        res
    }

    pub fn get_scope(&self) -> &Scope {
        unsafe { self.scopes.last().unwrap_unchecked() }
    }

    pub fn get_scope_mut(&mut self) -> &mut Scope {
        unsafe { self.scopes.last_mut().unwrap_unchecked() }
    }

    pub fn add_var(&mut self, ident: Ident, value: Value) {
        self.get_scope_mut().vars.insert(ident, value);
    }

    pub fn get_var(&self, ident: &Ident) -> Result<&Value, RuntimeError> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.vars.get(ident) {
                return Ok(value);
            }
        }

        Err(RuntimeError("Variable not defined.".into()))
    }

    pub fn get_var_mut(&mut self, ident: &Ident) -> Result<&mut Value, RuntimeError> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.vars.get_mut(ident) {
                return Ok(value);
            }
        }

        Err(RuntimeError("Variable not defined.".into()))
    }
}

#[derive(Clone, Debug)]
pub struct Scope {
    vars: HashMap<Ident, Value>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
}

// TODO: Wrap Rc on Value to avoid cloning
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
    fn evalulate(&self, ins: &mut Interpreter) -> Result<Value, RuntimeError>;
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
    fn evalulate_op(&self, ins: &mut Interpreter, op1: &Expr) -> Result<Value, RuntimeError> {
        let op1 = op1.evalulate(ins)?;

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
    fn evalulate_op(
        &self,
        ins: &mut Interpreter,
        op1: &Expr,
        op2: &Expr,
    ) -> Result<Value, RuntimeError> {
        // Assignment have special treatment
        if let BinaryOp::Assign = self {
            // We might need to handle this properly in the future (i.e. l-values), for now this will work

            let Expr::Variable(ident) = op1 else {
                return Err(RuntimeError(
                    "Ident expected on left hand side of assignment.".into(),
                ));
            };

            let val = op2.evalulate(ins)?;
            *ins.get_var_mut(ident)? = val.clone();

            return Ok(val);
        }

        let op1 = op1.evalulate(ins)?;
        let op2 = op2.evalulate(ins)?;

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
    fn evalulate(&self, ins: &mut Interpreter) -> Result<Value, RuntimeError> {
        Ok(match self {
            Expr::Literal(val) => match val {
                Literal::Null => Value::Null,
                Literal::Bool(val) => Value::Bool(*val),
                Literal::Number(val) => Value::Number(*val),
                Literal::String(val) => Value::String(val.clone()), // TODO: Don't use clone
            },
            Expr::Variable(ident) => ins.get_var(ident)?.clone(),
            Expr::FunctionCall(ident, expr) => {
                // TODO: REMOVE THIS
                if &*ident.0 == "print" {
                    println!("{:?}", (expr[0]).evalulate(ins)?);
                    Value::Null
                } else {
                    todo!()
                }
            }
            Expr::Grouped(expr) => expr.evalulate(ins)?,
            Expr::Unary(op, op_1) => op.evalulate_op(ins, op_1)?,
            Expr::Binary(op, op_1, op_2) => op.evalulate_op(ins, op_1, op_2)?,
        })
    }
}

impl Evalulate for Stmt {
    fn evalulate(&self, ins: &mut Interpreter) -> Result<Value, RuntimeError> {
        match self {
            Stmt::Expr(expr) => {
                expr.evalulate(ins)?;
            }
            Stmt::Block(block) => {
                block.evalulate(ins)?;
            }
            Stmt::Fn(_, _, _) => todo!(),
            Stmt::Return(_) => todo!(),
            Stmt::Let(ident, expr) => {
                let val = expr.evalulate(ins)?;
                ins.add_var(ident.clone(), val);
            }
            Stmt::If(branches, else_branch) => 'outer: {
                for (cond, branch) in branches.iter() {
                    if cond.evalulate(ins)?.is_truthy() {
                        branch.evalulate(ins)?;
                        break 'outer;
                    }
                }

                if let Some(else_branch) = else_branch {
                    else_branch.evalulate(ins)?;
                }
            }
        };

        Ok(Value::Null)
    }
}

impl Evalulate for Block {
    fn evalulate(&self, ins: &mut Interpreter) -> Result<Value, RuntimeError> {
        ins.eval_in_scope(|ins| {
            for stmt in self.0.iter() {
                stmt.evalulate(ins)?;
            }

            Ok(Value::Null)
        })
    }
}
