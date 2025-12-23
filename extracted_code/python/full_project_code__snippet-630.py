"""
Manages a set of fonts: normal, italic, bold, etc...
"""

def __init__(self, font_name, font_size=14):
    self.font_name = font_name
    self.font_size = font_size
    self.fonts = {}
    self.encoding = None
    if sys.platform.startswith('win'):
        if not font_name:
            self.font_name = DEFAULT_FONT_NAME_WIN
        self._create_win()
    elif sys.platform.startswith('darwin'):
        if not font_name:
            self.font_name = DEFAULT_FONT_NAME_MAC
        self._create_mac()
    else:
        if not font_name:
            self.font_name = DEFAULT_FONT_NAME_NIX
        self._create_nix()

def _get_nix_font_path(self, name, style):
    proc = subprocess.Popen(['fc-list', "%s:style=%s" % (name, style), 'file'],
                            stdout=subprocess.PIPE, stderr=None)
    stdout, _ = proc.communicate()
    if proc.returncode == 0:
        lines = stdout.splitlines()
        for line in lines:
            if line.startswith(b'Fontconfig warning:'):
                continue
            path = line.decode().strip().strip(':')
            if path:
                return path
        return None

def _create_nix(self):
    for name in STYLES['NORMAL']:
        path = self._get_nix_font_path(self.font_name, name)
        if path is not None:
            self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
            break
    else:
        raise FontNotFound('No usable fonts named: "%s"' %
                           self.font_name)
    for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
        for stylename in STYLES[style]:
            path = self._get_nix_font_path(self.font_name, stylename)
            if path is not None:
                self.fonts[style] = ImageFont.truetype(path, self.font_size)
                break
        else:
            if style == 'BOLDITALIC':
                self.fonts[style] = self.fonts['BOLD']
            else:
                self.fonts[style] = self.fonts['NORMAL']

def _get_mac_font_path(self, font_map, name, style):
    return font_map.get((name + ' ' + style).strip().lower())

def _create_mac(self):
    font_map = {}
    for font_dir in (os.path.join(os.getenv("HOME"), 'Library/Fonts/'),
                     '/Library/Fonts/', '/System/Library/Fonts/'):
        font_map.update(
            (os.path.splitext(f)[0].lower(), os.path.join(font_dir, f))
            for f in os.listdir(font_dir)
            if f.lower().endswith(('ttf', 'ttc')))

    for name in STYLES['NORMAL']:
        path = self._get_mac_font_path(font_map, self.font_name, name)
        if path is not None:
            self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
            break
    else:
        raise FontNotFound('No usable fonts named: "%s"' %
                           self.font_name)
    for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
        for stylename in STYLES[style]:
            path = self._get_mac_font_path(font_map, self.font_name, stylename)
            if path is not None:
                self.fonts[style] = ImageFont.truetype(path, self.font_size)
                break
        else:
            if style == 'BOLDITALIC':
                self.fonts[style] = self.fonts['BOLD']
            else:
                self.fonts[style] = self.fonts['NORMAL']

def _lookup_win(self, key, basename, styles, fail=False):
    for suffix in ('', ' (TrueType)'):
        for style in styles:
            try:
                valname = '%s%s%s' % (basename, style and ' '+style, suffix)
                val, _ = _winreg.QueryValueEx(key, valname)
                return val
            except OSError:
                continue
    else:
        if fail:
            raise FontNotFound('Font %s (%s) not found in registry' %
                               (basename, styles[0]))
        return None

def _create_win(self):
    lookuperror = None
    keynames = [ (_winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows NT\CurrentVersion\Fonts'),
                 (_winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Fonts'),
                 (_winreg.HKEY_LOCAL_MACHINE, r'Software\Microsoft\Windows NT\CurrentVersion\Fonts'),
                 (_winreg.HKEY_LOCAL_MACHINE, r'Software\Microsoft\Windows\CurrentVersion\Fonts') ]
    for keyname in keynames:
        try:
            key = _winreg.OpenKey(*keyname)
            try:
                path = self._lookup_win(key, self.font_name, STYLES['NORMAL'], True)
                self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
                for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
                    path = self._lookup_win(key, self.font_name, STYLES[style])
                    if path:
                        self.fonts[style] = ImageFont.truetype(path, self.font_size)
                    else:
                        if style == 'BOLDITALIC':
                            self.fonts[style] = self.fonts['BOLD']
                        else:
                            self.fonts[style] = self.fonts['NORMAL']
                return
            except FontNotFound as err:
                lookuperror = err
            finally:
                _winreg.CloseKey(key)
        except OSError:
            pass
    else:
        # If we get here, we checked all registry keys and had no luck
        # We can be in one of two situations now:
        # * All key lookups failed. In this case lookuperror is None and we
        #   will raise a generic error
        # * At least one lookup failed with a FontNotFound error. In this
        #   case, we will raise that as a more specific error
        if lookuperror:
            raise lookuperror
        raise FontNotFound('Can\'t open Windows font registry key')

def get_char_size(self):
    """
    Get the character size.
    """
    return self.get_text_size('M')

def get_text_size(self, text):
    """
    Get the text size (width, height).
    """
    font = self.fonts['NORMAL']
    if hasattr(font, 'getbbox'):  # Pillow >= 9.2.0
        return font.getbbox(text)[2:4]
    else:
        return font.getsize(text)

def get_font(self, bold, oblique):
    """
    Get the font based on bold and italic flags.
    """
    if bold and oblique:
        return self.fonts['BOLDITALIC']
    elif bold:
        return self.fonts['BOLD']
    elif oblique:
        return self.fonts['ITALIC']
    else:
        return self.fonts['NORMAL']


