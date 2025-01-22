import jamspell

jsp = jamspell.TSpellCorrector()
assert jsp.LoadLangModel('ru_small.bin')

def fix_spelling (word):
    fixed_word = jsp.FixFragment(word)
    return fixed_word