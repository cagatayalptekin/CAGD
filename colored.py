import os

os.system('color')

# danke https://stackoverflow.com/a/33206814/7622658

c_color = {
    "BLACK": 30,
    "RED": 31,
    "GREEN": 32,
    "YELLOW": 33,
    "BLUE": 34,
    "MAGENTA": 35,
    "CYAN": 36,
    "WHITE": 37,
    "B_BLACK": 90,
    "B_RED": 91,
    "B_GREEN": 92,
    "B_YELLOW": 93,
    "B_BLUE": 94,
    "B_MAGENTA": 95,
    "B_CYAN": 96,
    "B_WHITE": 97
}
c_style = {
    "BOLD": 1,
    "FAINT": 2,  # not working in windows, but on ubuntu
    "ITALIC": 3  # NOT WORKING
}
c_decoration = {
    "UNDERLINE": 4,
    "CROSSED": 9,  # not working in windows, but on ubuntu
    "OVERLINE": 53,  # NOT WORKING
    "FRAMED": 51,  # NOT WORKING
    "ENCIRCLED": 52  # NOT WORKING
}
c_blink = {
    "SLOW": 5,  # NOT WORKING
    "FAST": 6  # NOT WORKING
}


def colored(text, c=None, s=None, d=None, bg=None, b=None) -> str:
    # c=color / s=style / d=decoration / bg=background color / b=blink
    first = True
    modifier = "\033["

    if c is not None and c in c_color:
        modifier += str(c_color[c])
        first = False

    if s is not None and s in c_style:
        if not first:
            modifier += ";"
        modifier += str(c_style[s])
        first = False

    if d is not None and d in c_decoration:
        if not first:
            modifier += ";"
        modifier += str(c_decoration[d])
        first = False

    if bg is not None and bg in c_color:
        if not first:
            modifier += ";"
        modifier += str(c_color[bg] + 10)
        first = False

    if b is not None and b in c_blink:
        if not first:
            modifier += ";"
        modifier += str(c_blink[b])
        first = False

    if first:
        modifier += "0"

    modifier += "m"
    return f"{modifier}{text}\033[0m"
