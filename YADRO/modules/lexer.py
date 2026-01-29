#==========================================#
#  ==     =====   ==  ==   =====   ====    #
#  ==     =        ====    =       =  ==   #
#  ==     =====     ==     =====   = ==    #
#  ==     =        ====    =       =  ==   #
#  =====  =====   ==  ==   =====   =   ==  #
#==========================================#

#  Lexer module for YADRO compiler   

#  version 0.2.0 
#  Made by CyrOil

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union
#Tokens [a lot of vars to be eaten by Lexer]
class TokenType(Enum):
    DIRECTIVE_TARGET = "#target"
    DIRECTIVE_PLUGIN = "#plugin"
    DIRECTIVE_REQUIRES = "#requires"
    DIRECTIVE_IMPORT = "#import"
    DIRECTIVE_DEFINE = "#define"
    DIRECTIVE_START = "#start"
    DIRECTIVE_END = "#end"
    KEYWORD_FUN = "fun"
    KEYWORD_CLASS = "class"
    KEYWORD_TEMP = "temp"
    KEYWORD_IF = "if"
    KEYWORD_ELSE = "else"
    KEYWORD_ELSIF = "elsif"
    KEYWORD_SWITCH = "switch"
    KEYWORD_CASE = "case"
    KEYWORD_DEFAULT = "default"
    KEYWORD_FOR = "for"
    KEYWORD_WHILE = "while"
    KEYWORD_REPEAT = "repeat"
    KEYWORD_UNTIL = "until"
    KEYWORD_RETURN = "return"
    KEYWORD_DEL = "del"
    KEYWORD_CONST = "const"
    KEYWORD_IMPL = "impl"
    KEYWORD_TRAIT = "trait"
    KEYWORD_WHERE = "where"
    MODIFIER_ASYNC = "[async]"
    MODIFIER_THREAD = "[thread]"
    MODIFIER_CONST = "[const]"
    MODIFIER_GPU = "[gpu]"
    MODIFIER_CGPU = "[cgpu]"
    MODIFIER_CORO = "[coro]"
    MODIFIER_NONRET = "[nonret]"
    MODIFIER_LINEAR = "[linear]"
    MODIFIER_MODULE = "[module]"
    MODIFIER_ACTOR = "[actor]"
    MODIFIER_UNSAFE = "#[unsafe]"
    TYPE_INT = "int"
    TYPE_FLOAT = "float"
    TYPE_BOOL = "bool"
    TYPE_STRING = "string"
    TYPE_CHAR = "char"
    TYPE_VOID = "void"
    TYPE_UNIT = "Unit"
    TYPE_ARRAY = "array"
    TYPE_DARRAY = "darray"
    TYPE_DICT = "dict"
    TYPE_SET = "set"
    TYPE_VECTOR = "vector"
    TYPE_GC = "gc"
    TYPE_RESULT = "Result"
    TYPE_OPTION = "Option"
    TYPE_REF = "&"
    TYPE_REF_MUT = "&mut"
    OPERATOR_QUESTION = "?"
    KEYWORD_OK = "Ok"
    KEYWORD_ERR = "Err"
    KEYWORD_SOME = "Some"
    KEYWORD_NONE = "None"
    PREDICATE_TILDE = "~"
    OP_PIPELINE_FWD = ">>>"
    OP_PIPELINE_BWD = "<<<"
    OP_ASSIGN = "="
    OP_ASSIGN_ADD = "+="
    OP_ASSIGN_SUB = "-="
    OP_ASSIGN_MUL = "*="
    OP_ASSIGN_DIV = "/="
    OP_ASSIGN_MOD = "%="
    OP_ASSIGN_POW = "^="
    OP_ASSIGN_FDIV = "\\="
    OP_ASSIGN_BIT_OR = "|="
    OP_ASSIGN_BIT_AND = "&="
    OP_ASSIGN_BIT_XOR = "^="
    OP_ASSIGN_LSHIFT = "<<="
    OP_ASSIGN_RSHIFT = ">>="
    OP_ASSIGN_IF_GT = "=>"
    OP_ASSIGN_IF_LT = "=<"
    OP_ASSIGN_IF_GTE = "=>="
    OP_ASSIGN_IF_LTE = "=<="
    OP_ASSIGN_IF_NE = "=!="
    OP_ASSIGN_ADDR = "@="
    OP_ASSIGN_SWAP = "$="
    OP_EQ = "=="
    OP_NE = "!="
    OP_LT = "<"
    OP_GT = ">"
    OP_LTE = "<="
    OP_GTE = ">="
    OP_ADD = "+"
    OP_SUB = "-"
    OP_MUL = "*"
    OP_DIV = "/"
    OP_FDIV = "|"  
    OP_MOD = "%"
    OP_POW = "^"
    OP_ABS_START = "|a|" 
    OP_BIT_AND = "&"
    OP_BIT_OR = "|"
    OP_BIT_XOR = "^"
    OP_BIT_NOT = "~"
    OP_LSHIFT = "<<"
    OP_RSHIFT = ">>"
    OP_LOGICAL_AND = "and"
    OP_LOGICAL_OR = "or"
    OP_LOGICAL_XOR = "xor"
    OP_LOGICAL_NAND = "nand"
    OP_LOGICAL_ANY = "any"
    OP_LOGICAL_ALL = "all"
    OP_IMPLICATION = "->"
    OP_DEREF = "*"
    OP_LENGTH = "#"
    LITERAL_INT = "INT"
    LITERAL_FLOAT = "FLOAT"
    LITERAL_STRING = "STRING"
    LITERAL_CHAR = "CHAR"
    LITERAL_BOOL = "BOOL"
    LITERAL_HEX = "HEX"
    LITERAL_BIN = "BIN"
    LITERAL_OCT = "OCT"
    LITERAL_COMPLEX = "COMPLEX"
    IDENTIFIER = "IDENTIFIER"
    COLON = ":"
    SEMICOLON = ";"
    COMMA = ","
    DOT = "."
    DOUBLE_COLON = "::"
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    COMMENT_LINE = "//"
    COMMENT_BLOCK = "/* */"
    EOF = "EOF"
#Nice token info
@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str
    line: int
    column: int
    value: Optional[Union[int, float, str, bool]] = None
    def __str__(self):
        loc = f"{self.line}:{self.column}"
        val = f" val={self.value}" if self.value is not None else ""
        return f"Token({self.type.name:25} '{self.lexeme:25}' @ {loc}{val})"
#LexerError - is my impression of pessimist
class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"[Line {line}:{column}] {message}")
        self.line = line
        self.column = column
# Oh here we go again <Lexer is great guy gives tokens to lexems>
class YadroLexer:
    KEYWORDS = {
        "fun": TokenType.KEYWORD_FUN,
        "class": TokenType.KEYWORD_CLASS,
        "temp": TokenType.KEYWORD_TEMP,
        "if": TokenType.KEYWORD_IF,
        "else": TokenType.KEYWORD_ELSE,
        "elsif": TokenType.KEYWORD_ELSIF,
        "switch": TokenType.KEYWORD_SWITCH,
        "case": TokenType.KEYWORD_CASE,
        "default": TokenType.KEYWORD_DEFAULT,
        "for": TokenType.KEYWORD_FOR,
        "while": TokenType.KEYWORD_WHILE,
        "repeat": TokenType.KEYWORD_REPEAT,
        "until": TokenType.KEYWORD_UNTIL,
        "return": TokenType.KEYWORD_RETURN,
        "del": TokenType.KEYWORD_DEL,
        "const": TokenType.KEYWORD_CONST,
        "impl": TokenType.KEYWORD_IMPL,
        "trait": TokenType.KEYWORD_TRAIT,
        "where": TokenType.KEYWORD_WHERE,
        "int": TokenType.TYPE_INT,
        "float": TokenType.TYPE_FLOAT,
        "bool": TokenType.TYPE_BOOL,
        "string": TokenType.TYPE_STRING,
        "char": TokenType.TYPE_CHAR,
        "void": TokenType.TYPE_VOID,
        "Unit": TokenType.TYPE_UNIT,
        "array": TokenType.TYPE_ARRAY,
        "darray": TokenType.TYPE_DARRAY,
        "dict": TokenType.TYPE_DICT,
        "set": TokenType.TYPE_SET,
        "vector": TokenType.TYPE_VECTOR,
        "gc": TokenType.TYPE_GC,
        "Result": TokenType.TYPE_RESULT,
        "Option": TokenType.TYPE_OPTION,
        "Ok": TokenType.KEYWORD_OK,
        "Err": TokenType.KEYWORD_ERR,
        "Some": TokenType.KEYWORD_SOME,
        "None": TokenType.KEYWORD_NONE,
        "and": TokenType.OP_LOGICAL_AND,
        "or": TokenType.OP_LOGICAL_OR,
        "xor": TokenType.OP_LOGICAL_XOR,
        "nand": TokenType.OP_LOGICAL_NAND,
        "any": TokenType.OP_LOGICAL_ANY,
        "all": TokenType.OP_LOGICAL_ALL,
    }
    MODIFIERS = {
        "[async]": TokenType.MODIFIER_ASYNC,
        "[thread]": TokenType.MODIFIER_THREAD,
        "[const]": TokenType.MODIFIER_CONST,
        "[gpu]": TokenType.MODIFIER_GPU,
        "[cgpu]": TokenType.MODIFIER_CGPU,
        "[coro]": TokenType.MODIFIER_CORO,
        "[nonret]": TokenType.MODIFIER_NONRET,
        "[linear]": TokenType.MODIFIER_LINEAR,
        "[module]": TokenType.MODIFIER_MODULE,
        "[actor]": TokenType.MODIFIER_ACTOR,
    }
    DIRECTIVES = {
        "#target": TokenType.DIRECTIVE_TARGET,
        "#plugin": TokenType.DIRECTIVE_PLUGIN,
        "#requires": TokenType.DIRECTIVE_REQUIRES,
        "#import": TokenType.DIRECTIVE_IMPORT,
        "#define": TokenType.DIRECTIVE_DEFINE,
        "#start": TokenType.DIRECTIVE_START,
        "#end": TokenType.DIRECTIVE_END,
    }
    # clear it all
    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]
    # Ill give tokens for my fellow lexems
    def tokenize(self) -> List[Token]:
        try:
            while not self.is_at_end():
                self.start = self.current
                self.scan_token()
            self.add_token(TokenType.NEWLINE)
            while len(self.indent_stack) > 1:
                self.add_token(TokenType.DEDENT)
                self.indent_stack.pop()
            
            self.add_token(TokenType.EOF)
            return self.tokens
        except LexerError as e:
            print(f"LEXICAL ERROR ({self.filename}): {e}")
            raise
    
    def scan_token(self):
        c = self.advance()
        if c == ' ' or c == '\r':
            return
        elif c == '\n':
            self.add_token(TokenType.NEWLINE)
            self.column = 1
            self.line += 1
            self.scan_indentation()
            return
        elif c == '\t':
            raise LexerError("Tabs forbidden: use spaces for explicit indentation cost", 
                           self.line, self.column)
        elif c == '/' and self.match('/'):
            self.scan_line_comment()
            return
        elif c == '/' and self.match('*'):
            self.scan_block_comment()
            return
        elif c == '#':
            self.scan_directive()
            return
        elif c == '"' or c == "'":
            self.scan_string(c)
            return
        elif c.isdigit():
            self.scan_number()
            return
        elif c.isalpha() or c == '_':
            self.scan_identifier()
            return
        elif c == '>':
            if self.match('>'):
                if self.match('>'):
                    self.add_token(TokenType.OP_PIPELINE_FWD)  # >>>
                else:
                    self.add_token(TokenType.OP_RSHIFT)  # >>
            elif self.match('='):
                self.add_token(TokenType.OP_GTE)  # >=
            else:
                self.add_token(TokenType.OP_GT)  # >
                
        elif c == '<':
            if self.match('<'):
                if self.match('<'):
                    self.add_token(TokenType.OP_PIPELINE_BWD)  # <<<
                else:
                    self.add_token(TokenType.OP_LSHIFT)  # <<
            elif self.match('='):
                self.add_token(TokenType.OP_LTE)  # <=
            else:
                self.add_token(TokenType.OP_LT)  # <
                
        elif c == '=':
            if self.match('>'):
                if self.match('='):
                    self.add_token(TokenType.OP_ASSIGN_IF_GTE)  # =>=
                else:
                    self.add_token(TokenType.OP_ASSIGN_IF_GT)  # =>
            elif self.match('<'):
                if self.match('='):
                    self.add_token(TokenType.OP_ASSIGN_IF_LTE)  # =<=
                else:
                    self.add_token(TokenType.OP_ASSIGN_IF_LT)  # =<
            elif self.match('='):
                self.add_token(TokenType.OP_EQ)  # ==
            elif self.match('!'):
                if self.match('='):
                    self.add_token(TokenType.OP_ASSIGN_IF_NE)  # =!=
                else:
                    self.add_token(TokenType.OP_NE)  # !=
            else:
                self.add_token(TokenType.OP_ASSIGN)  # =
                
        elif c == '!':
            if self.match('='):
                self.add_token(TokenType.OP_NE)
            else:
                self.add_token(TokenType.OP_BIT_NOT)
                
        elif c == '+':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_ADD)
            else:
                self.add_token(TokenType.OP_ADD)
                
        elif c == '-':
            if self.match('>'):
                self.add_token(TokenType.OP_IMPLICATION)
            elif self.match('='):
                self.add_token(TokenType.OP_ASSIGN_SUB)
            else:
                self.add_token(TokenType.OP_SUB)
                
        elif c == '*':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_MUL)
            else:
                self.add_token(TokenType.OP_MUL)
                
        elif c == '/':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_DIV)
            else:
                self.add_token(TokenType.OP_DIV)
                
        elif c == '%':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_MOD)
            else:
                self.add_token(TokenType.OP_MOD)
                
        elif c == '^':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_POW)
            else:
                self.add_token(TokenType.OP_POW)
                
        elif c == '|':
            if self.match('a') and self.match('|'):
                self.add_token(TokenType.OP_ABS_START)
            elif self.match('='):
                self.add_token(TokenType.OP_ASSIGN_BIT_OR)
            else:
                self.add_token(TokenType.OP_FDIV)
                
        elif c == '&':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_BIT_AND)
            else:
                self.add_token(TokenType.OP_BIT_AND)
                
        elif c == '~':
            self.add_token(TokenType.PREDICATE_TILDE)
                
        elif c == '@':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_ADDR)
            else:
                raise LexerError("'@' alone forbidden: use explicit '&var' for references", 
                               self.line, self.column)
                
        elif c == '$':
            if self.match('='):
                self.add_token(TokenType.OP_ASSIGN_SWAP)
            else:
                raise LexerError("'$' alone forbidden: use '$=' for swap only", 
                               self.line, self.column)

        elif c == '[':
            if self.peek().isalpha():
                while self.peek().isalnum() or self.peek() == '_':
                    self.advance()
                if self.match(']'):
                    full_mod = self.source[self.start:self.current]
                    if full_mod in self.MODIFIERS:
                        self.add_token(self.MODIFIERS[full_mod])
                        return
                    else:
                        self.current = self.start + 1
                        self.column = self.column - (self.current - (self.start + 1))
                        self.add_token(TokenType.LBRACKET, "[")
                        self.scan_identifier()
                        if self.match(']'):
                            self.add_token(TokenType.RBRACKET, "]")
                        else:
                            raise LexerError(f"Expected ']' after modifier content", 
                                           self.line, self.column)
                else:
                    self.add_token(TokenType.LBRACKET, "[")
            else:
                self.add_token(TokenType.LBRACKET, "[")
                
        elif c == ']':
            self.add_token(TokenType.RBRACKET, "]")
        elif c == ':':
            if self.match(':'):
                self.add_token(TokenType.DOUBLE_COLON)
            else:
                self.add_token(TokenType.COLON)
        elif c == ';':
            self.add_token(TokenType.SEMICOLON)
        elif c == ',':
            self.add_token(TokenType.COMMA)
        elif c == '.':
            self.add_token(TokenType.DOT)
        elif c == '(':
            self.add_token(TokenType.LPAREN)
        elif c == ')':
            self.add_token(TokenType.RPAREN)
        elif c == '{':
            self.add_token(TokenType.LBRACE)
        elif c == '}':
            self.add_token(TokenType.RBRACE)
        elif c == '#':
            if self.peek().isalpha():
                if self.match('[') and self.match('unsafe') and self.match(']'):
                    self.add_token(TokenType.MODIFIER_UNSAFE)
                else:
                    self.add_token(TokenType.OP_LENGTH)
            else:
                self.add_token(TokenType.OP_LENGTH)
            
        else:
            raise LexerError(f"Unexpected character: '{c}'", self.line, self.column)
    
    def scan_directive(self):
        while self.peek().isalpha() or self.peek() == '_':
            self.advance()
        
        directive = self.source[self.start:self.current]
        
        if directive in self.DIRECTIVES:
            self.add_token(self.DIRECTIVES[directive])
        elif directive == "#[unsafe":
            if self.match(']'):
                self.add_token(TokenType.MODIFIER_UNSAFE)
            else:
                raise LexerError("Malformed #[unsafe] directive", self.line, self.column)
        else:
            raise LexerError(f"Unknown directive: {directive}", self.line, self.column)
    
    def scan_identifier(self):
        while self.is_alphanumeric(self.peek()):
            self.advance()
        
        text = self.source[self.start:self.current]
        token_type = self.KEYWORDS.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type, text)
    
    def scan_number(self):
        start_col = self.column - 1
        if self.source[self.start] == '0' and self.peek() in ('x', 'X'):
            self.advance()
            # oh its peek<PEAK>
            if not self.peek_hex():
                raise LexerError("Invalid hex literal: no digits after 0x", self.line, self.column)
            while self.peek_hex():
                self.advance()
            text = self.source[self.start:self.current]
            value = int(text, 16)
            self.add_token(TokenType.LITERAL_HEX, text, value)
            return
        
        if self.source[self.start] == '0' and self.peek() in ('b', 'B'):
            self.advance()
            if not self.peek_bin():
                raise LexerError("Invalid binary literal: no digits after 0b", self.line, self.column)
            while self.peek_bin():
                self.advance()
            text = self.source[self.start:self.current]
            value = int(text, 2)
            self.add_token(TokenType.LITERAL_BIN, text, value)
            return
        if self.source[self.start] == '0' and self.peek() in ('o', 'O'):
            self.advance()
            if not self.peek_oct():
                raise LexerError("Invalid octal literal: no digits after 0o", self.line, self.column)
            while self.peek_oct():
                self.advance()
            text = self.source[self.start:self.current]
            value = int(text, 8)
            self.add_token(TokenType.LITERAL_OCT, text, value)
            return
        while self.peek().isdigit():
            self.advance()
        is_float = False
        if self.peek() == '.' and self.peek_next().isdigit():
            is_float = True
            self.advance()
            while self.peek().isdigit():
                self.advance()
        if self.peek() in ('e', 'E'):
            is_float = True
            self.advance()
            if self.peek() in ('+', '-'):
                self.advance()
            if not self.peek().isdigit():
                raise LexerError("Invalid exponential notation: missing digits", 
                               self.line, self.column)
            while self.peek().isdigit():
                self.advance()
        if self.peek() in ('+', '-') and self.peek_next() == 'i':
            is_float = True  # Complex implies float
            self.advance()
            self.advance()  # consume 'i'
            text = self.source[self.start:self.current]
            self.add_token(TokenType.LITERAL_COMPLEX, text)
            return
        
        text = self.source[self.start:self.current]
        if is_float:
            self.add_token(TokenType.LITERAL_FLOAT, text, float(text))
        else:
            self.add_token(TokenType.LITERAL_INT, text, int(text))
    
    def scan_string(self, quote: str):
        is_multiline = False
        start_pos = self.current - 1
        
        if quote == "'" and self.match("''"):
            if self.match("'"):
                is_multiline = True
            else:
                self.add_token(TokenType.LITERAL_STRING, "''", "")
                return
        elif quote == '"' and self.match('"'):
            if self.match('"'):
                raise LexerError('Triple-double quotes forbidden: use \'\'\' for multiline', 
                               self.line, self.column)
        content_start = self.current
        while not self.is_at_end():
            if self.peek() == '\\':
                self.advance()
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 1
                self.advance()
                continue
            
            if is_multiline:
                if self.match("'''"):
                    break
            else:
                if self.peek() == quote:
                    self.advance()
                    break
            
            if self.peek() == '\n':
                if not is_multiline:
                    raise LexerError("Unterminated string literal", self.line, self.column)
                self.line += 1
                self.column = 1
            
            self.advance()
        else:
            raise LexerError("Unterminated string literal", self.line, self.column)
        content_end = self.current - (6 if is_multiline else 1)
        value = self.source[content_start:content_end]
        full_lexeme = self.source[start_pos:self.current]
        
        self.add_token(TokenType.LITERAL_STRING, full_lexeme, value)
    # okay you see me
    def scan_line_comment(self):
        while self.peek() != '\n' and not self.is_at_end():
            self.advance()
    def scan_block_comment(self):
        depth = 1
        while depth > 0 and not self.is_at_end():
            if self.peek() == '/' and self.peek_next() == '*':
                self.advance()
                self.advance()
                depth += 1
            elif self.peek() == '*' and self.peek_next() == '/':
                self.advance()
                self.advance()
                depth -= 1
            elif self.peek() == '\n':
                self.line += 1
                self.column = 1
                self.advance()
            else:
                self.advance()
        
        if depth > 0:
            raise LexerError("Unterminated block comment", self.line, self.column)
    
    def scan_indentation(self):
        spaces = 0
        while self.peek() == ' ':
            self.advance()
            spaces += 1
        if self.peek() == '\n' or self.is_at_end() or self.peek() == '#':
            return
        
        prev_indent = self.indent_stack[-1]
        if spaces > prev_indent:
            self.indent_stack.append(spaces)
            self.add_token(TokenType.INDENT)
        elif spaces < prev_indent:
            while self.indent_stack[-1] > spaces:
                self.indent_stack.pop()
                self.add_token(TokenType.DEDENT)
            if self.indent_stack[-1] != spaces:
                raise LexerError(
                    f"Inconsistent indentation: expected {self.indent_stack[-1]} spaces, got {spaces}",
                    self.line, self.column
                )
    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def advance(self) -> str:
        self.current += 1
        self.column += 1
        return self.source[self.current - 1]
    
    def match(self, expected: str) -> bool:
        if self.is_at_end():
            return False
        if not self.source.startswith(expected, self.current):
            return False
        self.current += len(expected)
        self.column += len(expected)
        return True
    
    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    #if you read this ILY
    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def peek_hex(self) -> bool:
        c = self.peek()
        return c.isdigit() or ('a' <= c.lower() <= 'f')
    
    def peek_bin(self) -> bool:
        return self.peek() in ('0', '1')
    
    def peek_oct(self) -> bool:
        c = self.peek()
        return '0' <= c <= '7'
    
    def is_alphanumeric(self, c: str) -> bool:
        return c.isalnum() or c == '_'
    
    def add_token(self, type: TokenType, lexeme: Optional[str] = None, value: Optional[Union[int, float, str]] = None):
        if lexeme is None:
            lexeme = self.source[self.start:self.current]
        token = Token(
            type=type,
            lexeme=lexeme,
            line=self.line,
            column=self.column - (self.current - self.start),
            value=value
        )
        self.tokens.append(token)