    # simple way to get the Latin alphabet pages from Wikipedia through
    # the API, so for now we just support Cyrillic.
    "Serbian": Language(
        name="Serbian",
        iso_code="sr",
        alphabet="АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШабвгдђежзијклљмнњопрстћуфхцчџш",
        charsets=["ISO-8859-5", "WINDOWS-1251", "MacCyrillic", "IBM855"],
        wiki_start_pages=["Главна_страна"],
    ),
    "Thai": Language(
        name="Thai",
        iso_code="th",
        use_ascii=False,
        charsets=["ISO-8859-11", "TIS-620", "CP874"],
        alphabet="กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุู฿เแโใไๅๆ็่้๊๋์ํ๎๏๐๑๒๓๔๕๖๗๘๙๚๛",
        wiki_start_pages=["หน้าหลัก"],
    ),
    "Turkish": Language(
        name="Turkish",
        iso_code="tr",
        # Q, W, and X are not used by Turkish
        use_ascii=False,
        charsets=["ISO-8859-3", "ISO-8859-9", "WINDOWS-1254"],
        alphabet="abcçdefgğhıijklmnoöprsştuüvyzâîûABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZÂÎÛ",
        wiki_start_pages=["Ana_Sayfa"],
    ),
    "Vietnamese": Language(
        name="Vietnamese",
        iso_code="vi",
        use_ascii=False,
        # Windows-1258 is the only common 8-bit
        # Vietnamese encoding supported by Python.
        # From Wikipedia:
        # For systems that lack support for Unicode,
        # dozens of 8-bit Vietnamese code pages are
        # available.[1] The most common are VISCII
        # (TCVN 5712:1993), VPS, and Windows-1258.[3]
        # Where ASCII is required, such as when
        # ensuring readability in plain text e-mail,
        # Vietnamese letters are often encoded
        # according to Vietnamese Quoted-Readable
        # (VIQR) or VSCII Mnemonic (VSCII-MNEM),[4]
        # though usage of either variable-width
        # scheme has declined dramatically following
        # the adoption of Unicode on the World Wide
        # Web.
        charsets=["WINDOWS-1258"],
        alphabet="aăâbcdđeêghiklmnoôơpqrstuưvxyAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY",
        wiki_start_pages=["Chữ_Quốc_ngữ"],
    ),
