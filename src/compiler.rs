use string_interner::{DefaultBackend, StringInterner};

use crate::{
    ast::{BinaryOp, Block, CompileError, Expr, FuncDecl, Stmt, UnaryOp},
    bytecode::{CodeObject, Instr, Value},
    instr,
    token::{Ident, Literal},
};

#[derive(Clone, Debug)]
pub struct CompileState {
    code: Vec<Instr>,
    stack: Vec<Option<Ident>>,
    consts: Vec<Literal>,
    global_names: Vec<Ident>,
    stack_count: u16,
    arg_count: u16,
}

impl CompileState {
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

    pub fn add_const(&mut self, lit: &Literal) -> u16 {
        // TODO: Cache literals
        let slot = self.consts.len() as u16;
        self.consts.push(lit.clone());

        slot
    }

    pub fn add_global(&mut self, ident: Ident) -> u16 {
        // TODO: Cache global
        let slot = self.global_names.len() as u16;
        self.global_names.push(ident);

        slot
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

    pub fn create_context(&self) -> usize {
        self.stack.len()
    }

    pub fn delete_context(&mut self, context: usize) {
        self.stack.truncate(context);
    }

    pub fn consume(self, interner: &StringInterner<DefaultBackend>) -> CodeObject {
        CodeObject {
            code: self.code.into(),
            consts: self
                .consts
                .iter()
                .map(|lit| literal_to_value(lit))
                .collect(),
            global_names: self
                .global_names
                .iter()
                .map(|ident| interner.resolve(ident.0).unwrap().into())
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
        Literal::String(_) => todo!(),
    }
}

pub fn compile_expr(state: &mut CompileState, expr: &Expr) -> Result<u16, CompileError> {
    match expr {
        Expr::Literal(lit) => {
            let const_slot = state.add_const(lit);
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

        Expr::Binary(op, g_expr_1, g_expr_2) => {
            let g_slot_1 = compile_expr(state, g_expr_1)?;
            let g_slot_2 = compile_expr(state, g_expr_2)?;

            if let BinaryOp::Assign = op {
                // TODO: Assignment is incomplete!
                state.add_instr(instr!(Copy, g_slot_1, g_slot_2));

                Ok(g_slot_1)
            } else {
                let slot = state.add_slot(None);

                state.add_instr(match op {
                    BinaryOp::Assign => unreachable!(),
                    BinaryOp::Add => instr!(OpAdd, slot, g_slot_1, g_slot_2),
                    BinaryOp::Minus => instr!(OpMinus, slot, g_slot_1, g_slot_2),
                    BinaryOp::Multiply => instr!(OpMultiply, slot, g_slot_1, g_slot_2),
                    BinaryOp::Divide => instr!(OpDivide, slot, g_slot_1, g_slot_2),
                    BinaryOp::Equal => instr!(CmpEqual, slot, g_slot_1, g_slot_2),
                    BinaryOp::NotEqual => instr!(CmpNotEqual, slot, g_slot_1, g_slot_2),
                    BinaryOp::Greater => instr!(CmpGreater, slot, g_slot_1, g_slot_2),
                    BinaryOp::Less => instr!(CmpLess, slot, g_slot_1, g_slot_2),
                    BinaryOp::GreaterOrEqual => instr!(CmpGreaterOrEqual, slot, g_slot_1, g_slot_2),
                    BinaryOp::LessOrEqual => instr!(CmpLessOrEqual, slot, g_slot_1, g_slot_2),
                    BinaryOp::And => instr!(OpAnd, slot, g_slot_1, g_slot_2),
                    BinaryOp::Or => instr!(OpOr, slot, g_slot_1, g_slot_2),
                });

                Ok(slot)
            }
        }

        Expr::Access(_, _) => todo!(),
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
        Stmt::Expr(expr) => {
            compile_expr(state, expr)?;
        }

        Stmt::Block(block) => {
            compile_block(state, block)?;
        }

        Stmt::Fn(_, _) => todo!(),

        Stmt::Let(ident, expr) => {
            let old_slot = compile_expr(state, expr)?;
            let slot = state.add_slot(Some(*ident));

            state.add_instr(instr!(Copy, slot, old_slot));
        }

        Stmt::If(if_chain, else_body) => {
            // TODO: Use labels!!!
            let mut jmp_instrs_at = vec![];

            for (cond, body) in if_chain.iter() {
                let cond_slot = compile_expr(state, cond)?;

                // Jump to the next branch
                let at = state.code.len();
                state.add_instr(instr!(Copy, 0));

                // Block
                compile_block(state, body)?;

                // Jump to the end
                jmp_instrs_at.push(state.code.len());
                state.add_instr(instr!(Copy, 0));

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

pub fn compile_fn(
    decl: &FuncDecl,
    interner: &StringInterner<DefaultBackend>,
) -> Result<CodeObject, CompileError> {
    let mut state = CompileState {
        code: vec![],
        stack: vec![],
        consts: vec![],
        global_names: vec![],
        stack_count: 0,
        arg_count: decl.parameters.len() as u16,
    };

    // Add parameters
    for par in decl.parameters.iter() {
        state.add_slot(Some(*par));
    }

    // Add statements
    for stmt in decl.block.0.iter() {
        compile_stmt(&mut state, stmt)?;
    }

    // TODO: Return null?
    let const_slot = state.add_const(&Literal::Null);
    let empty_slot = state.add_slot(None);
    state.add_instr(instr!(LoadConst, empty_slot, const_slot));
    state.add_instr(instr!(Return, empty_slot));

    // Get code object
    let code_object = state.consume(interner);

    Ok(code_object)
}

pub fn compile(
    stmts: &[Stmt],
    interner: &StringInterner<DefaultBackend>,
) -> Result<CodeObject, CompileError> {
    for stmt in stmts {
        match stmt {
            Stmt::Fn(_, decl) => {
                return compile_fn(decl, interner);
            }
            _ => panic!(),
        }
    }

    todo!()
}
