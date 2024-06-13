use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    ast::{BinaryOp, Block, ClassDecl, Expr, FunctionDecl, Stmt, UnaryOp},
    token::{Ident, Literal},
};

#[derive(Clone, Debug)]
pub enum Value<'cx> {
    Null,
    Bool(bool),
    Number(f64),
    String(Rc<str>),
    Function(Function<'cx>),
    Class(Rc<ClassDecl<'cx>>),
    Object(Rc<RefCell<Object<'cx>>>), // TODO: Use a GC
}

#[derive(Clone, Debug)]
pub enum Function<'cx> {
    Native(NativeFuncPtr),
    Program(Rc<FunctionDecl<'cx>>),
}

pub type NativeFuncPtr = for<'cx> fn(
    ins: &mut Interpreter<'cx>,
    values: Vec<Value<'cx>>,
) -> Result<Value<'cx>, ControlFlow<'cx>>;

#[derive(Clone, Debug)]
pub struct Interpreter<'cx> {
    scopes: Vec<Scope<'cx>>,
}

#[derive(Clone, Debug)]
pub enum ControlFlow<'cx> {
    Break,
    Continue,
    Return(Value<'cx>),
    Error(RuntimeError),
}

#[derive(Clone, Debug)]
pub struct RuntimeError(pub Box<str>);

macro_rules! runtime_error {
    ($msg: expr) => {
        return Err(ControlFlow::Error(RuntimeError($msg)))
    };
}

impl<'cx> Default for Interpreter<'cx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'cx> Interpreter<'cx> {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    pub fn eval_in_scope<Item>(
        &mut self,
        func: impl FnOnce(&mut Self) -> Result<Item, ControlFlow<'cx>>,
    ) -> Result<Item, ControlFlow<'cx>> {
        self.scopes.push(Scope::new());

        let res = func(self);

        self.scopes.pop();

        res
    }

    pub fn get_scope(&self) -> &Scope<'cx> {
        unsafe { self.scopes.last().unwrap_unchecked() }
    }

    pub fn get_scope_mut(&mut self) -> &mut Scope<'cx> {
        unsafe { self.scopes.last_mut().unwrap_unchecked() }
    }

    pub fn add_var(&mut self, ident: Ident<'cx>, value: Value<'cx>) {
        self.get_scope_mut().vars.insert(ident, value);
    }

    pub fn get_var(&self, ident: Ident<'cx>) -> Result<&Value<'cx>, ControlFlow<'cx>> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.vars.get(&ident) {
                return Ok(value);
            }
        }

        runtime_error!("Variable not defined.".into())
    }

    pub fn get_var_mut(&mut self, ident: Ident<'cx>) -> Result<&mut Value<'cx>, ControlFlow<'cx>> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.vars.get_mut(&ident) {
                return Ok(value);
            }
        }

        runtime_error!("Variable not defined.".into())
    }
}

#[derive(Clone, Debug)]
pub struct Scope<'cx> {
    vars: HashMap<Ident<'cx>, Value<'cx>>,
}

impl<'cx> Default for Scope<'cx> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'cx> Scope<'cx> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Object<'cx> {
    pub class: Rc<ClassDecl<'cx>>,
    pub fields: HashMap<Ident<'cx>, Value<'cx>>,
}

pub trait Evalulate<'cx>: Sized {
    type Item;

    fn evalulate(&self, ins: &mut Interpreter<'cx>) -> Result<Self::Item, ControlFlow<'cx>>;
}

impl<'cx> Value<'cx> {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(val) => *val,
            Value::Number(val) => *val != 0.0 && !(*val).is_nan(),
            Value::String(val) => !(*val).is_empty(),
            Value::Function(_) => true,
            Value::Class(_) => true,
            Value::Object(_) => true,
        }
    }

    pub fn is_equal(&self, rhs: &Self) -> bool {
        // Note: Make sure to update this table when a new value is added
        match (self, rhs) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(val_1), Value::Bool(val_2)) => val_1 == val_2,
            (Value::Number(val_1), Value::Number(val_2)) => val_1 == val_2,
            (Value::String(val_1), Value::String(val_2)) => val_1 == val_2,
            (
                Value::Function(Function::Native(func_1)),
                Value::Function(Function::Native(func_2)),
            ) => func_1.eq(func_2),
            (
                Value::Function(Function::Program(func_1)),
                Value::Function(Function::Program(func_2)),
            ) => Rc::ptr_eq(func_1, func_2),
            (Value::Class(class_1), Value::Class(class_2)) => Rc::ptr_eq(class_1, class_2),
            (Value::Object(obj_1), Value::Object(obj_2)) => Rc::ptr_eq(obj_1, obj_2),
            _ => false,
        }
    }
}

impl<'cx> UnaryOp {
    fn evalulate_op(
        &self,
        ins: &mut Interpreter<'cx>,
        op1: &Expr<'cx>,
    ) -> Result<Value<'cx>, ControlFlow<'cx>> {
        let op1 = op1.evalulate(ins)?;

        match self {
            UnaryOp::Negate => match op1 {
                Value::Number(val) => Ok(Value::Number(-val)),
                _ => runtime_error!("Negation requires number.".into()),
            },
            UnaryOp::Not => Ok(Value::Bool(!op1.is_truthy())),
        }
    }
}

impl<'cx> BinaryOp {
    fn evalulate_op(
        &self,
        ins: &mut Interpreter<'cx>,
        op1: &Expr<'cx>,
        op2: &Expr<'cx>,
    ) -> Result<Value<'cx>, ControlFlow<'cx>> {
        // Assignment have special treatment
        if let BinaryOp::Assign = self {
            // We might need to handle this properly in the future (i.e. l-values), for now this will work

            let val = op2.evalulate(ins)?;

            match op1 {
                Expr::Variable(ident) => {
                    *ins.get_var_mut(*ident)? = val.clone();
                }
                Expr::Access(expr, ident) => {
                    let Value::Object(obj) = expr.evalulate(ins)? else {
                        runtime_error!("Only . operator can be used on object.".into());
                    };

                    obj.borrow_mut().fields.insert(*ident, val.clone());
                }
                _ => runtime_error!("Invalid left hand side of assignment.".into()),
            };

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
                    runtime_error!("Expected number.".into())
                };
                let Value::Number(val_2) = op2 else {
                    runtime_error!("Expected number.".into())
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

impl<'cx> Evalulate<'cx> for Expr<'cx> {
    type Item = Value<'cx>;

    fn evalulate(&self, ins: &mut Interpreter<'cx>) -> Result<Self::Item, ControlFlow<'cx>> {
        Ok(match self {
            Expr::Literal(val) => match val {
                Literal::Null => Value::Null,
                Literal::Bool(val) => Value::Bool(*val),
                Literal::Number(val) => Value::Number(*val),
                Literal::String(val) => Value::String(Rc::clone(val)),
            },
            Expr::Variable(ident) => ins.get_var(*ident)?.clone(),
            Expr::FunctionCall(func_expr, exprs) => {
                match func_expr.evalulate(ins)? {
                    // Function
                    Value::Function(func) => {
                        let mut values = Vec::with_capacity(exprs.len());
                        for expr in exprs.iter() {
                            values.push(expr.evalulate(ins)?);
                        }

                        match func {
                            Function::Native(func) => {
                                // It shouldn't be necessary to create a scope before calling a native function?
                                func(ins, values)?
                            }
                            Function::Program(func) => {
                                let mut values = values.into_iter();

                                ins.eval_in_scope(|ins| {
                                    for i in 0..func.parameters.len() {
                                        ins.add_var(
                                            func.parameters[i],
                                            values.next().unwrap_or(Value::Null),
                                        );
                                    }

                                    let res = func.block.evalulate(ins);
                                    match res {
                                        Err(ControlFlow::Return(ret_val)) => return Ok(ret_val),
                                        Err(ControlFlow::Break | ControlFlow::Continue) => {
                                            // Technically this should be compile error
                                            runtime_error!(
                                                "Invalid control flow outside loop.".into()
                                            )
                                        }
                                        _ => res?,
                                    }

                                    // TODO: Return
                                    Ok(Value::Null)
                                })?
                            }
                        }
                    }
                    // Class
                    Value::Class(class) => {
                        let obj = Object {
                            class,
                            fields: HashMap::new(),
                        };

                        Value::Object(Rc::new(RefCell::new(obj)))
                    }
                    _ => runtime_error!("Invalid function call.".into()),
                }
            }
            Expr::Grouped(expr) => expr.evalulate(ins)?,
            Expr::Unary(op, op_1) => op.evalulate_op(ins, op_1)?,
            Expr::Binary(op, op_1, op_2) => op.evalulate_op(ins, op_1, op_2)?,
            Expr::Access(expr, ident) => {
                let Value::Object(obj) = expr.evalulate(ins)? else {
                    runtime_error!("Only . operator can be used on object.".into());
                };

                let temp = obj.borrow();
                let Some(val) = temp.fields.get(ident) else {
                    runtime_error!("Unknown field.".into());
                };

                val.clone()
            }
        })
    }
}

impl<'cx> Evalulate<'cx> for Stmt<'cx> {
    type Item = ();

    fn evalulate(&self, ins: &mut Interpreter<'cx>) -> Result<Self::Item, ControlFlow<'cx>> {
        match self {
            Stmt::Expr(expr) => {
                expr.evalulate(ins)?;
            }
            Stmt::Block(block) => {
                block.evalulate(ins)?;
            }
            Stmt::Fn(ident, func) => {
                ins.add_var(*ident, Value::Function(Function::Program(Rc::clone(func))));
            }
            Stmt::Let(ident, expr) => {
                let val = expr.evalulate(ins)?;
                ins.add_var(*ident, val);
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
            Stmt::While(expr, block) => {
                while expr.evalulate(ins)?.is_truthy() {
                    let res = block.evalulate(ins);
                    match res {
                        Err(ControlFlow::Break) => break,
                        Err(ControlFlow::Continue) => continue,
                        _ => res?,
                    }
                }
            }
            Stmt::Break => return Err(ControlFlow::Break),
            Stmt::Continue => return Err(ControlFlow::Continue),
            Stmt::Return(expr) => return Err(ControlFlow::Return(expr.evalulate(ins)?)),
            Stmt::Class(class) => {
                ins.add_var(class.ident, Value::Class(Rc::new(class.clone())));
            }
        };

        Ok(())
    }
}

impl<'cx> Evalulate<'cx> for Block<'cx> {
    type Item = ();

    fn evalulate(&self, ins: &mut Interpreter<'cx>) -> Result<Self::Item, ControlFlow<'cx>> {
        ins.eval_in_scope(|ins| {
            for stmt in self.0.iter() {
                stmt.evalulate(ins)?;
            }

            Ok(())
        })
    }
}
