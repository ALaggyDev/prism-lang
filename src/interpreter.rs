use std::collections::HashMap;

use gc::{Finalize, Gc, GcCell, Trace};
use string_interner::{DefaultBackend, StringInterner};

use crate::{
    ast::{BinaryOp, Block, ClassDecl, Expr, FunctionDecl, Stmt, UnaryOp},
    native_func::NATIVE_FUNCS,
    token::{Ident, Literal},
};

#[derive(Clone, Debug, Trace, Finalize)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(Gc<String>),
    Callable(Callable),
    Class(Gc<ClassDecl>),
    Object(Gc<GcCell<Object>>),
}

#[derive(Clone, Trace, Finalize)]
pub enum Callable {
    Native(#[unsafe_ignore_trace] NativeFuncPtr),
    Function(Gc<FunctionDecl>),
    Method(Box<(Value, Gc<FunctionDecl>)>),
}

pub type NativeFuncPtr =
    fn(ins: &mut Interpreter, values: Vec<Value>) -> Result<Value, ControlFlow>;

#[derive(Clone, Debug)]
pub struct Interpreter {
    scopes: Vec<Scope>,
    interner: StringInterner<DefaultBackend>,
}

#[derive(Clone, Debug)]
pub enum ControlFlow {
    Break,
    Continue,
    Return(Value),
    Error(RuntimeError),
}

#[derive(Clone, Debug)]
pub struct RuntimeError(pub Box<str>);

macro_rules! runtime_error {
    ($msg: expr) => {
        return Err(ControlFlow::Error(RuntimeError($msg)))
    };
}

impl Interpreter {
    pub fn new(interner: StringInterner<DefaultBackend>) -> Self {
        let mut ins = Self {
            scopes: vec![Scope::new()],
            interner,
        };

        for (name, func) in NATIVE_FUNCS.iter() {
            let ident = Ident(ins.interner.get_or_intern_static(name));
            ins.add_var(ident, Value::Callable(Callable::Native(*func)));
        }

        ins
    }

    pub fn eval_in_scope<Item>(
        &mut self,
        func: impl FnOnce(&mut Self) -> Result<Item, ControlFlow>,
    ) -> Result<Item, ControlFlow> {
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

    pub fn get_var(&self, ident: Ident) -> Result<&Value, ControlFlow> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.vars.get(&ident) {
                return Ok(value);
            }
        }

        runtime_error!("Variable not defined.".into())
    }

    pub fn get_var_mut(&mut self, ident: Ident) -> Result<&mut Value, ControlFlow> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.vars.get_mut(&ident) {
                return Ok(value);
            }
        }

        runtime_error!("Variable not defined.".into())
    }

    pub fn get_this(&self) -> Result<Value, ControlFlow> {
        for scope in self.scopes.iter().rev() {
            if let Some(this) = &scope.this {
                return Ok(this.clone());
            }
        }

        runtime_error!("Keyword `this` is not set.".into());
    }
}

#[derive(Clone, Debug)]
pub struct Scope {
    this: Option<Value>,
    vars: HashMap<Ident, Value>,
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Scope {
    pub fn new() -> Self {
        Self {
            this: None,
            vars: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Trace, Finalize)]
pub struct Object {
    pub class: Gc<ClassDecl>,
    pub fields: HashMap<Ident, Value>,
}

pub trait Evalulate: Sized {
    type Item;

    fn evalulate(&self, ins: &mut Interpreter) -> Result<Self::Item, ControlFlow>;
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(val) => *val,
            Value::Number(val) => *val != 0.0 && !(*val).is_nan(),
            Value::String(val) => !(*val).is_empty(),
            Value::Callable(_) => true,
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
                Value::Callable(Callable::Native(func_1)),
                Value::Callable(Callable::Native(func_2)),
            ) => func_1.eq(func_2),
            (
                Value::Callable(Callable::Function(func_1)),
                Value::Callable(Callable::Function(func_2)),
            ) => Gc::ptr_eq(func_1, func_2),
            (Value::Class(class_1), Value::Class(class_2)) => Gc::ptr_eq(class_1, class_2),
            (Value::Object(obj_1), Value::Object(obj_2)) => Gc::ptr_eq(obj_1, obj_2),
            _ => false,
        }
    }
}

impl UnaryOp {
    fn evalulate_op(&self, ins: &mut Interpreter, op1: &Expr) -> Result<Value, ControlFlow> {
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

impl BinaryOp {
    fn evalulate_op(
        &self,
        ins: &mut Interpreter,
        op1: &Expr,
        op2: &Expr,
    ) -> Result<Value, ControlFlow> {
        // Assignment have special treatment
        if let BinaryOp::Assign = self {
            // We might need to handle this properly in the future (i.e. l-values), for now this will work

            let val = op2.evalulate(ins)?;

            match op1 {
                Expr::Variable(ident) => {
                    *ins.get_var_mut(*ident)? = val.clone();
                }
                Expr::Access(expr, ident) => {
                    let Value::Object(ref obj) = expr.evalulate(ins)? else {
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

fn eval_fn(
    ins: &mut Interpreter,
    func: &FunctionDecl,
    values: Vec<Value>,
    extra_fn: impl FnOnce(&mut Interpreter),
) -> Result<Value, ControlFlow> {
    let mut values = values.into_iter();

    ins.eval_in_scope(|ins| {
        extra_fn(ins);

        // Bind parameters
        for i in 0..func.parameters.len() {
            ins.add_var(func.parameters[i], values.next().unwrap_or(Value::Null));
        }

        let res = func.block.evalulate(ins);
        match res {
            Err(ControlFlow::Return(ret_val)) => return Ok(ret_val),
            Err(ControlFlow::Break | ControlFlow::Continue) => {
                // Technically this should be compile error
                runtime_error!("Invalid control flow outside loop.".into())
            }
            _ => res?,
        }

        // TODO: Return
        Ok(Value::Null)
    })
}

impl Evalulate for Expr {
    type Item = Value;

    fn evalulate(&self, ins: &mut Interpreter) -> Result<Self::Item, ControlFlow> {
        Ok(match self {
            Expr::Literal(val) => match val {
                Literal::Null => Value::Null,
                Literal::Bool(val) => Value::Bool(*val),
                Literal::Number(val) => Value::Number(*val),
                Literal::String(val) => Value::String(Gc::new(val.to_string())),
            },
            Expr::Variable(ident) => ins.get_var(*ident)?.clone(),
            Expr::FunctionCall(call_expr, exprs) => {
                match call_expr.evalulate(ins)? {
                    // Function
                    Value::Callable(ref callable) => {
                        let mut values = Vec::with_capacity(exprs.len());
                        for expr in exprs.iter() {
                            values.push(expr.evalulate(ins)?);
                        }

                        match callable {
                            Callable::Native(func) => {
                                // It shouldn't be necessary to create a scope before calling a native function?
                                func(ins, values)?
                            }
                            Callable::Function(func) => eval_fn(ins, &func, values, |_| {})?,
                            Callable::Method(method) => eval_fn(ins, &method.1, values, |ins| {
                                ins.get_scope_mut().this = Some(method.0.clone());
                            })?,
                        }
                    }
                    // Class
                    Value::Class(ref class) => {
                        let obj = Gc::new(GcCell::new(Object {
                            class: Gc::clone(&class),
                            fields: HashMap::new(),
                        }));

                        // Bind methods
                        for method in class.methods.iter() {
                            obj.borrow_mut().fields.insert(
                                method.0,
                                Value::Callable(Callable::Method(Box::new((
                                    Value::Object(obj.clone()),
                                    Gc::clone(&method.1),
                                )))),
                            );
                        }

                        // Call initializer
                        for method in class.methods.iter() {
                            if method.0 == Ident(ins.interner.get_or_intern_static("__init__")) {
                                eval_fn(ins, &method.1, Vec::new(), |ins| {
                                    ins.get_scope_mut().this = Some(Value::Object(obj.clone()));
                                })?;
                                break;
                            }
                        }

                        Value::Object(obj)
                    }
                    _ => runtime_error!("Invalid function call.".into()),
                }
            }
            Expr::Grouped(expr) => expr.evalulate(ins)?,
            Expr::Unary(op, op_1) => op.evalulate_op(ins, op_1)?,
            Expr::Binary(op, op_1, op_2) => op.evalulate_op(ins, op_1, op_2)?,
            Expr::Access(expr, ident) => {
                let Value::Object(ref obj) = expr.evalulate(ins)? else {
                    runtime_error!("Only . operator can be used on object.".into());
                };

                let temp = obj.borrow();
                let Some(val) = temp.fields.get(ident) else {
                    runtime_error!("Unknown field.".into());
                };

                val.clone()
            }
            Expr::This => ins.get_this()?,
        })
    }
}

impl Evalulate for Stmt {
    type Item = ();

    fn evalulate(&self, ins: &mut Interpreter) -> Result<Self::Item, ControlFlow> {
        match self {
            Stmt::Expr(expr) => {
                expr.evalulate(ins)?;
            }
            Stmt::Block(block) => {
                block.evalulate(ins)?;
            }
            Stmt::Fn(ident, func) => {
                ins.add_var(*ident, Value::Callable(Callable::Function(Gc::clone(func))));
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
                ins.add_var(class.ident, Value::Class(Gc::new(class.clone())));
            }
        };

        Ok(())
    }
}

impl Evalulate for Block {
    type Item = ();

    fn evalulate(&self, ins: &mut Interpreter) -> Result<Self::Item, ControlFlow> {
        ins.eval_in_scope(|ins| {
            for stmt in self.0.iter() {
                stmt.evalulate(ins)?;
            }

            Ok(())
        })
    }
}

// Temp hack to stop println!() from entering a loop
impl std::fmt::Debug for Callable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Callable::Native(f0) => f.debug_tuple("Native").field(&f0).finish(),
            Callable::Function(f0) => f.debug_tuple("Function").field(&f0).finish(),
            Callable::Method(f0) => f.debug_tuple("Method").field(&f0.1).finish(),
        }
    }
}
