use std::{ops::Range, slice};

use gc_arena::{Collect, Collection, Gc};

use crate::{
    bytecode::Instr,
    vm::{CodeObject, Value},
};

#[derive(Clone, Debug)]
pub struct CallFrame<'gc> {
    pub code_obj: Gc<'gc, CodeObject<'gc>>,
    stack: (*mut Value<'gc>, usize),
    pub ip: u16,
    pub ret_slot: u16,
}

unsafe impl<'gc> Collect for CallFrame<'gc> {
    #[inline]
    fn needs_trace() -> bool {
        true
    }
    #[inline]
    fn trace(&self, cc: &Collection) {
        self.code_obj.trace(cc)
    }
}

#[derive(Clone, Debug, Collect)]
#[collect(no_drop)]
pub struct CallFrames<'gc> {
    frames: Vec<CallFrame<'gc>>,
    stack: Vec<Value<'gc>>,
}

impl<'gc> CallFrame<'gc> {
    pub fn fetch_instr(&self) -> Option<Instr> {
        self.code_obj.code.get(self.ip as usize).copied()
    }

    pub fn get_slot(&self, index: u16) -> &Value<'gc> {
        &self.stack_slice()[index as usize]
    }

    pub fn get_mut_slot(&mut self, index: u16) -> &mut Value<'gc> {
        &mut self.stack_slice_mut()[index as usize]
    }

    pub fn stack_slice(&self) -> &[Value<'gc>] {
        unsafe { slice::from_raw_parts(self.stack.0, self.stack.1) }
    }

    pub fn stack_slice_mut(&mut self) -> &mut [Value<'gc>] {
        unsafe { slice::from_raw_parts_mut(self.stack.0, self.stack.1) }
    }
}

impl<'gc> CallFrames<'gc> {
    pub fn new() -> Self {
        Self {
            frames: vec![],
            stack: vec![],
        }
    }

    pub fn push_frame(&mut self, code_obj: Gc<'gc, CodeObject<'gc>>, args: &[Value<'gc>]) {
        if code_obj.arg_count as usize != args.len() {
            panic!("Argument count mismatch!");
        }

        let frame_stack = self.alloc(code_obj.stack_count as usize);

        let mut frame = CallFrame {
            code_obj,
            stack: (frame_stack, code_obj.stack_count as usize),
            ip: 0,
            ret_slot: 0,
        };

        frame.stack_slice_mut()[..args.len()].copy_from_slice(args);

        self.frames.push(frame);
    }

    pub(crate) fn push_frame_within(
        &mut self,
        code_obj: Gc<'gc, CodeObject<'gc>>,
        args: Range<usize>,
    ) {
        if code_obj.arg_count as usize != args.len() {
            panic!("Argument count mismatch!");
        }

        let old_offset =
            unsafe { self.get_last().stack.0.offset_from(self.stack.as_ptr()) as usize };

        let frame_stack = self.alloc(code_obj.stack_count as usize);

        let frame = CallFrame {
            code_obj,
            stack: (frame_stack, code_obj.stack_count as usize),
            ip: 0,
            ret_slot: 0,
        };

        let new_offset = unsafe { frame_stack.offset_from(self.stack.as_ptr()) as usize };
        self.stack.copy_within(
            (old_offset + args.start)..(old_offset + args.end),
            new_offset,
        );

        self.frames.push(frame);
    }

    pub fn pop_frame(&mut self) {
        let frame = self.frames.pop().unwrap();

        self.dealloc(frame.stack.1);
    }

    pub fn get_last(&self) -> &CallFrame<'gc> {
        self.frames.last().unwrap()
    }

    pub fn get_last_mut(&mut self) -> &mut CallFrame<'gc> {
        self.frames.last_mut().unwrap()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    fn alloc(&mut self, len: usize) -> *mut Value<'gc> {
        let old_ptr = self.stack.as_mut_ptr();

        let i = self.stack.len();
        for _ in 0..len {
            self.stack.push(Value::Null);
        }

        let new_ptr = self.stack.as_mut_ptr();

        if old_ptr != new_ptr {
            // Vec has reallocated, we need to rewrite each pointers to the Vec
            self.rewrite_ptr(new_ptr);
        }

        unsafe { new_ptr.offset(i as isize) }
    }

    fn dealloc(&mut self, len: usize) {
        // NOTE: We may need to reduce the capacity
        self.stack.truncate(self.stack.len() - len);
    }

    fn rewrite_ptr(&mut self, mut ptr: *mut Value<'gc>) {
        for frame in &mut self.frames {
            frame.stack.0 = ptr;
            ptr = unsafe { ptr.offset(frame.stack.1 as isize) };
        }
    }
}
