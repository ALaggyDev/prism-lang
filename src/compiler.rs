use gc::Gc;
use string_interner::{DefaultBackend, StringInterner};

use crate::{
    ast::{BinaryOp, Block, CompileError, Expr, FuncDecl, Stmt, UnaryOp},
    bytecode::{Callable, CodeObject, Instr, Value},
    instr,
    token::{Ident, Literal},
};

#[derive(Clone, Debug)]
pub struct CompileState<'a> {
    code: Vec<Instr>,
    stack: Vec<Option<Ident>>,
    consts: Vec<Value>,
    global_names: Vec<Ident>,
    stack_count: u16,
    arg_count: u16,

    is_global: bool,
    context_count: usize,

    interner: &'a StringInterner<DefaultBackend>,
}

impl<'a> CompileState<'a> {
    pub fn add_instr(&mut self, instr: Instr) {
        self.code.push(instr);
    }

    pub fn add_slot(&mut self, ident: Option<Ident>) -> u16 {
        let slot = self.stack.len() as u16;
        self.stack.push(ident);

        if slot + 1 > self.stack_count {
            self.stack_count = slot + 1;
        }

        slot
    }

    pub fn add_const(&mut self, value: Value) -> u16 {
        // Find if we have already added the value (NOTE: Should Value::cmp_equal be used?)
        if let Some(slot) = self.consts.iter().position(|val| val.cmp_equal(&value)) {
            slot as u16
        } else {
            // Else, add the value
            let slot = self.consts.len() as u16;
            self.consts.push(value);

            slot
        }
    }

    pub fn add_global(&mut self, ident: Ident) -> u16 {
        // Find if we have already added the value
        if let Some(slot) = self.global_names.iter().position(|val| *val == ident) {
            slot as u16
        } else {
            // Else, add the value
            let slot = self.global_names.len() as u16;
            self.global_names.push(ident);

            slot
        }
    }

    pub fn find_ident_in_stack(&self, ident: &Ident) -> Option<u16> {
        for (i, id) in self.stack.iter().enumerate().rev() {
            if let Some(id) = id {
                if id == ident {
                    return Some(i as u16);
                }
            }
        }

        None
    }

    pub fn create_context(&mut self) -> usize {
        self.context_count += 1;
        self.stack.len()
    }

    pub fn delete_context(&mut self, context: usize) {
        self.context_count -= 1;
        self.stack.truncate(context);
    }

    pub fn at_global(&self) -> bool {
        self.is_global && self.context_count == 0
    }

    pub fn consume(self) -> CodeObject {
        CodeObject {
            code: self.code.into(),
            consts: self.consts.into(),
            global_names: self
                .global_names
                .iter()
                .map(|ident| self.interner.resolve(ident.0).unwrap().into())
                .collect(),
            stack_count: self.stack_count,
            arg_count: self.arg_count,
        }
    }
}

pub fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Bool(val) => Value::Bool(*val),
        Literal::Number(val) => Value::Number(*val),
        Literal::String(val) => Value::String(Gc::new(val.clone())),
    }
}

pub fn handle_lvalue(
    state: &mut CompileState,
    l_expr: &Expr,
    r_expr: &Expr,
) -> Result<u16, CompileError> {
    let r_expr = compile_expr(state, r_expr)?;

    match l_expr {
        Expr::Variable(ident) => {
            if let Some(slot) = state.find_ident_in_stack(ident) {
                // Write to a local variable
                state.add_instr(instr!(Copy, slot, r_expr));
            } else if state.at_global() {
                // Else, if we can write into global, write into global
                let global_slot = state.add_global(*ident);

                state.add_instr(instr!(StoreGlobal, global_slot, r_expr));
            } else {
                panic!("Variable not found.")
            }
        }
        Expr::Index(expr, index) => {
            let expr_slot = compile_expr(state, expr)?;
            let index_slot = compile_expr(state, index)?;

            state.add_instr(instr!(StoreIndex, expr_slot, index_slot, r_expr));
        }
        _ => panic!("Not a l-value."),
    };

    Ok(r_expr)
}

pub fn compile_expr(state: &mut CompileState, expr: &Expr) -> Result<u16, CompileError> {
    match expr {
        Expr::Literal(lit) => {
            let const_slot = state.add_const(literal_to_value(lit));
            let slot = state.add_slot(None);

            state.add_instr(instr!(LoadConst, slot, const_slot));

            Ok(slot)
        }

        Expr::Variable(ident) => {
            if let Some(slot) = state.find_ident_in_stack(ident) {
                // Use local variable
                Ok(slot)
            } else {
                // Else, use global variable
                let slot = state.add_slot(None);
                let global_slot = state.add_global(*ident);

                state.add_instr(instr!(LoadGlobal, slot, global_slot));
                Ok(slot)
            }
        }

        Expr::FunctionCall(func_expr, arg_exprs) => {
            // Get the function
            let func_slot = compile_expr(state, func_expr)?;

            // Get the arguments
            let old_arg_slots: Box<[_]> = arg_exprs
                .iter()
                .map(|arg| compile_expr(state, arg))
                .collect::<Result<_, CompileError>>()?;

            // Copy the arguments into consecutive slots
            let ret_slot = state.add_slot(None);

            for &old_arg_slot in old_arg_slots.iter() {
                let new_arg_slot = state.add_slot(None);
                state.add_instr(instr!(Copy, new_arg_slot, old_arg_slot));
            }

            // Call the function
            state.add_instr(instr!(Call, func_slot, ret_slot, arg_exprs.len() as u16));

            Ok(ret_slot)
        }

        Expr::Grouped(g_expr) => compile_expr(state, g_expr),

        Expr::Unary(op, g_expr) => {
            let g_slot = compile_expr(state, g_expr)?;
            let slot = state.add_slot(None);

            state.add_instr(match op {
                UnaryOp::Negate => instr!(OpNegate, slot, g_slot),
                UnaryOp::Not => instr!(OpNot, slot, g_slot),
            });

            Ok(slot)
        }

        Expr::Binary(BinaryOp::Assign, l_expr, r_expr) => handle_lvalue(state, l_expr, r_expr),

        Expr::Binary(op, l_expr, r_expr) => {
            let l_expr = compile_expr(state, l_expr)?;
            let r_expr = compile_expr(state, r_expr)?;

            let slot = state.add_slot(None);

            state.add_instr(match op {
                BinaryOp::Assign => unreachable!(),
                BinaryOp::Add => instr!(OpAdd, slot, l_expr, r_expr),
                BinaryOp::Minus => instr!(OpMinus, slot, l_expr, r_expr),
                BinaryOp::Multiply => instr!(OpMultiply, slot, l_expr, r_expr),
                BinaryOp::Divide => instr!(OpDivide, slot, l_expr, r_expr),
                BinaryOp::Equal => instr!(CmpEqual, slot, l_expr, r_expr),
                BinaryOp::NotEqual => instr!(CmpNotEqual, slot, l_expr, r_expr),
                BinaryOp::Greater => instr!(CmpGreater, slot, l_expr, r_expr),
                BinaryOp::Less => instr!(CmpLess, slot, l_expr, r_expr),
                BinaryOp::GreaterOrEqual => instr!(CmpGreaterOrEqual, slot, l_expr, r_expr),
                BinaryOp::LessOrEqual => instr!(CmpLessOrEqual, slot, l_expr, r_expr),
                BinaryOp::And => instr!(OpAnd, slot, l_expr, r_expr),
                BinaryOp::Or => instr!(OpOr, slot, l_expr, r_expr),
            });

            Ok(slot)
        }

        Expr::Tuple(exprs) => {
            let old_expr_slots: Box<[_]> = exprs
                .iter()
                .map(|expr| compile_expr(state, expr))
                .collect::<Result<_, CompileError>>()?;

            let tuple_slot = state.add_slot(None);

            // Copy the arguments into consecutive slots
            for &old_expr_slot in old_expr_slots.iter() {
                let new_expr_slot = state.add_slot(None);
                state.add_instr(instr!(Copy, new_expr_slot, old_expr_slot));
            }

            // Pack tuple
            state.add_instr(instr!(
                PackTuple,
                tuple_slot,
                tuple_slot + 1,
                exprs.len() as u16
            ));

            Ok(tuple_slot)
        }

        Expr::Array(exprs) => {
            let old_expr_slots: Box<[_]> = exprs
                .iter()
                .map(|expr| compile_expr(state, expr))
                .collect::<Result<_, CompileError>>()?;

            let array_slot = state.add_slot(None);

            // Copy the arguments into consecutive slots
            for &old_expr_slot in old_expr_slots.iter() {
                let new_expr_slot = state.add_slot(None);
                state.add_instr(instr!(Copy, new_expr_slot, old_expr_slot));
            }

            // Pack array
            state.add_instr(instr!(
                PackArray,
                array_slot,
                array_slot + 1,
                exprs.len() as u16
            ));

            Ok(array_slot)
        }

        Expr::Index(expr_1, expr_2) => {
            let expr_1_slot = compile_expr(state, expr_1)?;
            let expr_2_slot = compile_expr(state, expr_2)?;

            let slot = state.add_slot(None);

            state.add_instr(instr!(LoadIndex, slot, expr_1_slot, expr_2_slot));

            Ok(slot)
        }
    }
}

pub fn compile_block(state: &mut CompileState, block: &Block) -> Result<(), CompileError> {
    let context = state.create_context();

    for g_stmt in block.0.iter() {
        compile_stmt(state, g_stmt)?;
    }

    state.delete_context(context);
    Ok(())
}

pub fn compile_stmt(state: &mut CompileState, stmt: &Stmt) -> Result<(), CompileError> {
    match stmt {
        Stmt::Empty => {}

        Stmt::Expr(expr) => {
            compile_expr(state, expr)?;
        }

        Stmt::Block(block) => {
            compile_block(state, block)?;
        }

        Stmt::Fn(ident, func_decl) => {
            let code_object = compile_fn(state, func_decl)?;
            let code_slot = state.add_const(Value::Callable(Callable::Func(Gc::new(code_object))));

            if !state.at_global() {
                // Normal context
                let slot = state.add_slot(Some(*ident));
                state.add_instr(instr!(LoadConst, slot, code_slot));
            } else {
                // If we are at global context, we use StoreGlobal

                let slot = state.add_slot(None);
                state.add_instr(instr!(LoadConst, slot, code_slot));

                let global_slot = state.add_global(*ident);
                state.add_instr(instr!(StoreGlobal, global_slot, slot));
            }
        }

        Stmt::Let(ident, expr) => {
            let old_slot = compile_expr(state, expr)?;

            if !state.at_global() {
                // Normal context
                let slot = state.add_slot(Some(*ident));

                state.add_instr(instr!(Copy, slot, old_slot));
            } else {
                // If we are at global context, we use StoreGlobal instead of Copy

                let global_slot = state.add_global(*ident);
                state.add_instr(instr!(StoreGlobal, global_slot, old_slot));
            }
        }

        Stmt::If(if_chain, else_body) => {
            // TODO: Use labels!!!
            // Apparently, this dumb way of emitting jump instruction is called `backpatching`, anyway IMPLEMENT LABELS!!!
            let mut jmp_instrs_at = vec![];

            for (i, (cond, body)) in if_chain.iter().enumerate() {
                let cond_slot = compile_expr(state, cond)?;

                // Jump to the next branch
                let at = state.code.len();
                state.add_instr(instr!(Copy, 0));

                // Block
                compile_block(state, body)?;

                // Jump to the end
                // If we are at the last if chain and there's no else body, we skip emitting jump instruction
                if !(i + 1 == if_chain.len() && else_body.is_none()) {
                    jmp_instrs_at.push(state.code.len());
                    state.add_instr(instr!(Copy, 0));
                }

                state.code[at] = instr!(JumpNotIf, state.code.len() as u16, cond_slot);
            }

            if let Some(else_body) = else_body {
                compile_block(state, else_body)?;
            }

            let end_at = state.code.len();
            for instr_at in jmp_instrs_at {
                state.code[instr_at] = instr!(Jump, end_at as u16);
            }
        }

        Stmt::While(cond, block) => {
            let at = state.code.len();

            // Jump out if false
            let cond_slot = compile_expr(state, cond)?;
            let temp_at = state.code.len();
            state.add_instr(instr!(Copy, 0));

            // Block
            compile_block(state, block)?;

            // Jump back
            state.add_instr(instr!(Jump, at as u16));

            let end_at = state.code.len();
            state.code[temp_at] = instr!(JumpNotIf, end_at as u16, cond_slot);
        }

        Stmt::Break => todo!(),
        Stmt::Continue => todo!(),

        Stmt::Return(expr) => {
            let expr_slot = compile_expr(state, expr)?;

            state.add_instr(instr!(Return, expr_slot));
        }

        Stmt::Class(_) => todo!(),
    }

    Ok(())
}

pub fn compile_fn(state: &mut CompileState, decl: &FuncDecl) -> Result<CodeObject, CompileError> {
    let mut state = CompileState {
        code: vec![],
        stack: vec![],
        consts: vec![],
        global_names: vec![],
        stack_count: 0,
        arg_count: decl.parameters.len() as u16,

        is_global: false,
        context_count: 0,

        interner: state.interner,
    };

    // Add parameters
    for par in decl.parameters.iter() {
        state.add_slot(Some(*par));
    }

    // Add statements
    for stmt in decl.block.0.iter() {
        compile_stmt(&mut state, stmt)?;
    }

    // Return null
    let const_slot = state.add_const(Value::Null);
    let empty_slot = state.add_slot(None);
    state.add_instr(instr!(LoadConst, empty_slot, const_slot));
    state.add_instr(instr!(Return, empty_slot));

    // Get code object
    let code_object = state.consume();

    Ok(code_object)
}

/// Compile a program. The result is a code object representing the global context.
pub fn compile(
    stmts: &[Stmt],
    interner: &StringInterner<DefaultBackend>,
) -> Result<CodeObject, CompileError> {
    let mut state = CompileState {
        code: vec![],
        stack: vec![],
        consts: vec![],
        global_names: vec![],
        stack_count: 0,
        arg_count: 0,

        is_global: true,
        context_count: 0,

        interner,
    };

    // Add statements
    for stmt in stmts {
        compile_stmt(&mut state, stmt)?;
    }

    // Get code object
    let code_object = state.consume();

    Ok(code_object)
}
