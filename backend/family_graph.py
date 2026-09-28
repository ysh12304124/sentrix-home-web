RELATION_INVERSES = {
    "丈夫": {"妻子"},
    "妻子": {"丈夫"},
    "父亲": {"儿子", "女儿"},
    "母亲": {"儿子", "女儿"},
    "儿子": {"父亲", "母亲"},
    "女儿": {"父亲", "母亲"},
    "哥哥": {"哥哥", "姐姐", "弟弟", "妹妹"},
    "姐姐": {"哥哥", "姐姐", "弟弟", "妹妹"},
    "弟弟": {"哥哥", "姐姐", "弟弟", "妹妹"},
    "妹妹": {"哥哥", "姐姐", "弟弟", "妹妹"},
    "祖父": {"孙子", "孙女"},
    "祖母": {"孙子", "孙女"},
    "外祖父": {"外孙", "外孙女"},
    "外祖母": {"外孙", "外孙女"},
    "孙子": {"祖父", "祖母"},
    "孙女": {"祖父", "祖母"},
    "外孙": {"外祖父", "外祖母"},
    "外孙女": {"外祖父", "外祖母"},
    "叔叔": {"侄子", "侄女"},
    "伯伯": {"侄子", "侄女"},
    "姑姑": {"侄子", "侄女"},
    "舅舅": {"外甥", "外甥女"},
    "姨妈": {"外甥", "外甥女"},
    "姨母": {"外甥", "外甥女"},
    "侄子": {"叔叔", "伯伯", "姑姑"},
    "侄女": {"叔叔", "伯伯", "姑姑"},
    "外甥": {"舅舅", "姨妈", "姨母"},
    "外甥女": {"舅舅", "姨妈", "姨母"},
    "朋友": {"朋友"},
    "密友": {"密友"},
    "其他亲属": {"其他亲属"},
}


def validate_relationship_pair(predicate, inverse_predicate):
    predicate = str(predicate or "").strip()
    inverse_predicate = str(inverse_predicate or "").strip()
    if not predicate or not inverse_predicate:
        raise ValueError("关系标签不能为空")
    allowed = RELATION_INVERSES.get(predicate)
    if not allowed or inverse_predicate not in allowed:
        raise ValueError(f"关系标签不匹配: {predicate} -> {inverse_predicate}")
