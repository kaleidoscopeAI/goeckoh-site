def __init__(self, fg=None, bg=None, bold=False, underline=False, italic=False):
    self.fg = fg
    self.bg = bg
    self.bold = bold
    self.underline = underline
    self.italic = italic

def escape(self, attrs):
    if len(attrs):
        return "\x1b[" + ";".join(attrs) + "m"
    return ""

def color_string(self):
    attrs = []
    if self.fg is not None:
        if self.fg in ansicolors:
            esc = codes[self.fg.replace('ansi','')]
            if ';01m' in esc:
                self.bold = True
            # extract fg color code.
            attrs.append(esc[2:4])
        else:
            attrs.extend(("38", "5", "%i" % self.fg))
    if self.bg is not None:
        if self.bg in ansicolors:
            esc = codes[self.bg.replace('ansi','')]
            # extract fg color code, add 10 for bg.
            attrs.append(str(int(esc[2:4])+10))
        else:
            attrs.extend(("48", "5", "%i" % self.bg))
    if self.bold:
        attrs.append("01")
    if self.underline:
        attrs.append("04")
    if self.italic:
        attrs.append("03")
    return self.escape(attrs)

def true_color_string(self):
    attrs = []
    if self.fg:
        attrs.extend(("38", "2", str(self.fg[0]), str(self.fg[1]), str(self.fg[2])))
    if self.bg:
        attrs.extend(("48", "2", str(self.bg[0]), str(self.bg[1]), str(self.bg[2])))
    if self.bold:
        attrs.append("01")
    if self.underline:
        attrs.append("04")
    if self.italic:
        attrs.append("03")
    return self.escape(attrs)

def reset_string(self):
    attrs = []
    if self.fg is not None:
        attrs.append("39")
    if self.bg is not None:
        attrs.append("49")
    if self.bold or self.underline or self.italic:
        attrs.append("00")
    return self.escape(attrs)


