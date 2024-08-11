use std::fmt;

use gc_arena::Collect;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum InstrKind {
    Copy,

    LoadConst,
    LoadGlobal,
    StoreGlobal,

    LoadIndex,
    StoreIndex,

    OpAdd,
    OpMinus,
    OpMultiply,
    OpDivide,

    OpNegate,

    OpAnd,
    OpOr,
    OpNot,

    CmpEqual,
    CmpNotEqual,
    CmpGreater,
    CmpLess,
    CmpGreaterOrEqual,
    CmpLessOrEqual,

    Jump,
    JumpIf,
    JumpNotIf,

    PackTuple,
    UnpackTuple,

    PackArray,

    // PackDict,
    Call,
    Return,
}

#[derive(Copy, Clone, PartialEq, Eq, Collect)]
#[collect(require_static)]
pub struct Instr(u64);

impl Instr {
    pub fn new_1(kind: InstrKind, a1: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16))
    }

    pub fn new_2(kind: InstrKind, a1: u16, a2: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16) | ((a2 as u64) << 32))
    }

    pub fn new_3(kind: InstrKind, a1: u16, a2: u16, a3: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16) | ((a2 as u64) << 32) | ((a3 as u64) << 48))
    }

    pub fn kind(self) -> InstrKind {
        unsafe { std::mem::transmute(self.0 as u16) }
    }

    pub fn a1(self) -> u16 {
        (self.0 >> 16) as u16
    }

    pub fn a2(self) -> u16 {
        (self.0 >> 32) as u16
    }

    pub fn a3(self) -> u16 {
        (self.0 >> 48) as u16
    }
}

impl fmt::Debug for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}({}, {}, {})",
            self.kind(),
            self.a1(),
            self.a2(),
            self.a3()
        )
    }
}

#[macro_export]
macro_rules! instr {
    ($kind: ident, $a1: expr) => {
        $crate::bytecode::Instr::new_1($crate::bytecode::InstrKind::$kind, $a1)
    };
    ($kind: ident, $a1: expr, $a2: expr) => {
        $crate::bytecode::Instr::new_2($crate::bytecode::InstrKind::$kind, $a1, $a2)
    };
    ($kind: ident, $a1: expr, $a2: expr, $a3: expr) => {
        $crate::bytecode::Instr::new_3($crate::bytecode::InstrKind::$kind, $a1, $a2, $a3)
    };
}
