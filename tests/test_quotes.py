from catalist.quotes import best_quote, is_exact

TURN = "Oh yeah. I think we should lose the L_C_D_ screen, like you said. It consumes batteries like hell."


def test_exact_quote_ignores_case_and_spacing():
    assert is_exact("i think we should   LOSE the L_C_D_ screen", TURN)


def test_short_or_paraphrased_quote_is_not_exact():
    assert not is_exact("lose the L_C_D_ screen", TURN)  # 4 words
    assert not is_exact("I think we should drop the LCD screen", TURN)  # paraphrase


def test_exact_quote_is_kept_as_is():
    quote = "we should lose the L_C_D_ screen"
    assert best_quote(quote, TURN) == quote


def test_paraphrase_is_replaced_by_closest_real_words():
    quote = best_quote("we should drop the LCD screen because of batteries", TURN)
    assert is_exact(quote, TURN)
    assert "lose the L_C_D_ screen" in quote


def test_turn_too_short_to_quote():
    assert best_quote("yes okay", "Yes, okay.") is None
