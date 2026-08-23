"""Grapheme-cluster-aware string reversal.

reverse_string(s) reverses a string by user-perceived characters
(grapheme clusters, Unicode UAX #29) instead of raw code points, so
combining marks, ZWJ emoji sequences, variation selectors, skin-tone
modifiers, keycap sequences, CRLF, flag pairs and Hangul syllables
stay glued to their base character and never get scrambled.
"""

import unicodedata
from typing import Optional

_ZWJ = '\u200d'                       # zero-width joiner (emoji sequences)
_RI_START, _RI_END = 0x1F1E6, 0x1F1FF  # regional indicators (flag pairs)
_TAG_START, _TAG_END = 0xE0020, 0xE007F  # emoji tag sequence characters
_MOD_START, _MOD_END = 0x1F3FB, 0x1F3FF  # emoji skin-tone modifiers (GB11)
_KEYCAP = 0x20E3                         # combining enclosing keycap
_VS_START, _VS_END = 0xFE00, 0xFE0F        # variation selectors
_VS_SUP_START, _VS_SUP_END = 0xE0100, 0xE01EF  # supplementary variation selectors
_L_START, _L_END = 0x1100, 0x115F   # Hangul leading jamo (L)
_V_START, _V_END = 0x1160, 0x11A7   # Hangul vowel jamo (V)
_T_START, _T_END = 0x11A8, 0x11FF   # Hangul trailing jamo (T)
_LVT_START, _LVT_END = 0xAC00, 0xD7A3  # precomposed Hangul syllables (LV/LVT)


def _hangul_class(cp: int) -> Optional[str]:
    """Classify a code point as Hangul L/V/T/LV/LVT, or None."""
    if _L_START <= cp <= _L_END:
        return 'L'
    if _V_START <= cp <= _V_END:
        return 'V'
    if _T_START <= cp <= _T_END:
        return 'T'
    if _LVT_START <= cp <= _LVT_END:
        # LV syllables have a zero trailing jamo (cp - base) % 28 == 0.
        return 'LV' if (cp - _LVT_START) % 28 == 0 else 'LVT'
    return None


def _trailing_ri_count(cluster: str) -> int:
    """Count regional indicators at the end of `cluster`."""
    n = 0
    for c in reversed(cluster):
        if _RI_START <= ord(c) <= _RI_END:
            n += 1
        else:
            break
    return n


def _attaches_to_previous(prev: str, ch: str) -> bool:
    """Return True if code point `ch` must stay glued to the previous cluster."""
    cp = ord(ch)
    pc = ord(prev[-1])

    # CRLF is a single grapheme (GB3).
    if ch == '\n' and prev.endswith('\r'):
        return True
    # Combining / extending marks stick to their base character (GB9).
    if unicodedata.combining(ch) != 0:
        return True
    # Spacing marks, e.g. Devanagari/Thai vowel signs (GB9b).
    if unicodedata.category(ch) == 'Mc':
        return True
    # Variation selectors select a glyph variant (GB9): FE00–FE0F + E0100–E01EF.
    if _VS_START <= cp <= _VS_END or _VS_SUP_START <= cp <= _VS_SUP_END:
        return True
    # A ZWJ continues the sequence (GB9), and the code point after ZWJ
    # continues it too (GB11), e.g. 👨‍👩‍👧‍👦.
    if ch == _ZWJ or prev.endswith(_ZWJ):
        return True
    # Emoji skin-tone modifiers stay on their base emoji (GB11).
    if _MOD_START <= cp <= _MOD_END:
        return True
    # An enclosing keycap completes a keycap sequence (GB10), "1️⃣".
    if cp == _KEYCAP:
        return True
    # Regional indicators pair up from the start of the run (GB12/GB13):
    # attach only when an odd number of RIs precedes this one, so
    # "🇦🇧🇨" clusters as "🇦🇧" + "🇨" and reverses to "🇨🇦🇧".
    if _RI_START <= cp <= _RI_END:
        return _trailing_ri_count(prev) % 2 == 1
    # Tag characters belong to an emoji tag sequence (GB9c).
    if _TAG_START <= cp <= _TAG_END:
        return True
    # Hangul jamo (GB6/GB7):
    #   L  × (L | V | LV | LVT)
    #   (LV | V) × (V | T)
    #   (LVT | T) × T
    p = _hangul_class(pc)
    c = _hangul_class(cp)
    if p is not None and c is not None:
        if p == 'L' and c in ('L', 'V', 'LV', 'LVT'):
            return True
        if p in ('LV', 'V') and c in ('V', 'T'):
            return True
        if p in ('LVT', 'T') and c == 'T':
            return True
    return False


def reverse_string(s: str) -> str:
    """Reverse `s` by grapheme clusters. Empty input returns ''."""
    if not s:
        return s

    clusters: list[str] = []
    for ch in s:
        if clusters and _attaches_to_previous(clusters[-1], ch):
            clusters[-1] += ch
        else:
            clusters.append(ch)

    return ''.join(reversed(clusters))


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # console may default to GBK/cp936
    except Exception:
        pass

    cases = [
        ("", ""),                                  # empty string
        ("hello", "olleh"),                        # ASCII
        ("你好世界", "界世好你"),                    # CJK
        ("e\u0301", "e\u0301"),                    # e + combining acute: one grapheme
        ("a\u0301b", "ba\u0301"),                  # combining mark stays on its base
        ("👨\u200d👩\u200d👧\u200d👦", "👨\u200d👩\u200d👧\u200d👦"),  # ZWJ family emoji
        ("a👨\u200d👩\u200d👧\u200d👦b", "b👨\u200d👩\u200d👧\u200d👦a"),
        ("🇨🇳x", "x🇨🇳"),                          # flag = regional-indicator pair
        ("🇦🇧🇨", "🇨🇦🇧"),                          # 3 RIs: pairs form from the start
        ("a🇦🇧🇨🇩b", "b🇨🇩🇦🇧a"),                  # 4 RIs: two flag pairs
        ("葛\ufe00", "葛\ufe00"),                  # CJK + variation selector (FE00)
        ("葛\U000e0100", "葛\U000e0100"),          # CJK + supplementary VS (E0100)
        ("\U0001f44d\U0001f3fdx", "x\U0001f44d\U0001f3fd"),  # 👍 + skin tone (GB11)
        ("1\ufe0f\u20e3", "1\ufe0f\u20e3"),       # keycap sequence stays whole
        ("a\r\nb", "b\r\na"),                      # CRLF is one grapheme (GB3)
        ("\u0915\u093e", "\u0915\u093e"),          # Devanagari क + vowel sign (Mc)
        ("a\u0915\u093eb", "b\u0915\u093ea"),      # spacing mark stays on its base
        ("\u1100\u1161", "\u1100\u1161"),          # decomposed Hangul 가 (L+V)
        ("a\u1100\u1161b", "b\u1100\u1161a"),      # Hangul syllable stays whole
        ("\uac01\u11a8", "\uac01\u11a8"),          # 각 + trailing jamo (LVT+T)
    ]
    ok = True
    for inp, expected in cases:
        got = reverse_string(inp)
        passed = got == expected
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {inp!r:>34} -> {got!r}")
    print('\nAll tests passed!' if ok else '\nSome tests FAILED')
