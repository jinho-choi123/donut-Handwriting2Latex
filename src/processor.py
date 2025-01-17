from transformers import ViTImageProcessor, BertTokenizer
from .config import config

PRETRAINED_DECODER_REPO_ID = config.get("PRETRAINED_DECODER_REPO_ID", "google-bert/bert-base-cased")
PRETRAINED_ENCODER_REPO_ID = config.get("PRETRAINED_ENCODER_REPO_ID", "google/vit-base-patch16-224-in21k")

custom_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_DECODER_REPO_ID)

new_tokens = [
    "<eos>",
    "<stroke>",
    "!",
    "&",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "¡",
    "=",
    "¿",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    r"\#",
    r"\%",
    r"\&",
    r"\Delta",
    r"\Gamma",
    r"\Lambda",
    r"\Leftrightarrow",
    r"\Omega",
    r"\Phi",
    r"\Pi",
    r"\Psi",
    r"\Rightarrow",
    r"\Sigma",
    r"\Theta",
    r"\Upsilon",
    r"\Vdash",
    r"\Xi",
    r"\ ",
    r"\aleph",
    r"\alpha",
    r"\angle",
    r"\approx",
    r"\backslash",
    r"\beginmatrix",
    r"\beta",
    r"\bigcap",
    r"\bigcirc",
    r"\bigcup",
    r"\bigoplus",
    r"\bigvee",
    r"\bigwedge",
    r"\bullet",
    r"\cap",
    r"\cdot",
    r"\chi",
    r"\circ",
    r"\cong",
    r"\cup",
    r"\dagger",
    r"\delta",
    r"\div",
    r"\dot",
    r"\emptyset",
    r"\endmatrix",
    r"\epsilon",
    r"\equiv",
    r"\eta",
    r"\exists",
    r"\forall",
    r"\frac",
    r"\gamma",
    r"\ge",
    r"\gg",
    r"\hat",
    r"\hbar",
    r"\hookrightarrow",
    r"\iff",
    r"\iint",
    r"\in",
    r"\infty",
    r"\int",
    r"\iota",
    r"\kappa",
    r"\lambda",
    r"\langle",
    r"\lceil",
    r"\le",
    r"\leftarrow",
    r"\leftrightarrow",
    r"\lfloor",
    r"\ll",
    r"\longrightarrow",
    r"\mapsto",
    r"\mathbb",
    r"\models",
    r"\mp",
    r"\mu",
    r"\nabla",
    r"\ne",
    r"\neg",
    r"\ni",
    r"\not",
    r"\notin",
    r"\nu",
    r"\odot",
    r"\oint",
    r"\omega",
    r"\ominus",
    r"\oplus",
    r"\otimes",
    r"\overline",
    r"\partial",
    r"\perp",
    r"\phi",
    r"\pi",
    r"\pm",
    r"\prime",
    r"\prod",
    r"\propto",
    r"\psi",
    r"\rangle",
    r"\rceil",
    r"\rfloor",
    r"\rho",
    r"\rightarrow",
    r"\rightleftharpoons",
    r"\sigma",
    r"\sim",
    r"\simeq",
    r"\sqrt",
    r"\sqsubseteq",
    r"\subset",
    r"\subseteq",
    r"\subsetneq",
    r"\sum",
    r"\supset",
    r"\supseteq",
    r"\tau",
    r"\theta",
    r"\tilde",
    r"\times",
    r"\top",
    r"\triangle",
    r"\triangleleft",
    r"\triangleq",
    r"\underline",
    r"\upsilon",
    r"\varphi",
    r"\varpi",
    r"\varsigma",
    r"\vartheta",
    r"\vdash",
    r"\vdots",
    r"\vec",
    r"\vee",
    r"\wedge",
    r"\xi",
    r"\zeta",
    r"\{",
    r"\—",
    r"\}",
    "]",
    "ˆ",
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "\\",
]

new_tokens = set(new_tokens) - set(custom_tokenizer.vocab.keys())
custom_tokenizer.add_tokens(list(new_tokens))

custom_image_processor = ViTImageProcessor.from_pretrained(PRETRAINED_ENCODER_REPO_ID)

