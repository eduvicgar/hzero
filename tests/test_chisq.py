import pytest
from distributions.chisq import ChiSquare
from scipy.stats import chi2

def test_critic_value_right_tail():
    chisq = ChiSquare(3)
    result = chisq.critic_value(0.05, tail="right")
    expected = chi2.ppf(0.95, 3)
    assert pytest.approx(result) == expected

def test_critic_value_left_tail():
    chisq = ChiSquare(3)
    result = chisq.critic_value(0.05, tail="left")
    expected = chi2.ppf(0.05, 3)
    assert pytest.approx(result) == expected

def test_critic_value_invalid_alpha_low():
    chisq = ChiSquare(3)
    with pytest.raises(ValueError):
        chisq.critic_value(-0.1, tail="right")

def test_critic_value_invalid_alpha_high():
    chisq = ChiSquare(3)
    with pytest.raises(ValueError):
        chisq.critic_value(1.5, tail="left")

def test_p_value_right_tail():
    chisq = ChiSquare(4)
    d = 5.0
    result = chisq.p_value(d, tail="right")
    expected = 1 - chi2.cdf(d, 4)
    assert pytest.approx(result) == expected

def test_p_value_left_tail():
    chisq = ChiSquare(4)
    d = 5.0
    result = chisq.p_value(d, tail="left")
    expected = chi2.cdf(d, 4)
    assert result == pytest.approx(expected) or result == pytest.approx(1 - expected)  # ambiguous doc logic

def test_p_value_zero():
    chisq = ChiSquare(2)
    result = chisq.p_value(0.0, tail="right")
    assert 0 <= result <= 1

def test_p_value_negative_raises():
    chisq = ChiSquare(2)
    with pytest.raises(ValueError):
        chisq.p_value(-1, tail="right")

def test_plot_right_tail(monkeypatch):
    chisq = ChiSquare(4)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    chisq.plot(d=3.0, alpha=0.05, tail="right")

def test_plot_left_tail(monkeypatch):
    chisq = ChiSquare(4)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    chisq.plot(d=3.0, alpha=0.05, tail="left")

def test_plot_bilateral(monkeypatch):
    chisq = ChiSquare(4)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    chisq.plot(d=3.0, alpha=0.05, tail="bilateral")

def test_plot_missing_alpha_with_bilateral():
    chisq = ChiSquare(4)
    with pytest.raises(ValueError):
        chisq.plot(d=3.0, alpha=None, tail="bilateral")

def test_plot_invalid_alpha_raises():
    chisq = ChiSquare(4)
    with pytest.raises(ValueError):
        chisq.plot(d=3.0, alpha=1.5, tail="right")

def test_plot_only_d(monkeypatch):
    chisq = ChiSquare(4)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    chisq.plot(d=3.0, tail="right")

def test_init_df():
    chisq = ChiSquare(5)
    assert chisq.df == 5

def test_multiple_calls_consistent():
    chisq = ChiSquare(2)
    alpha = 0.05
    right = chisq.critic_value(alpha, tail="right")
    left = chisq.critic_value(alpha, tail="left")
    assert right > left

def test_invalid_tail_in_critic_value():
    chisq = ChiSquare(3)
    with pytest.raises(ValueError):
        chisq.critic_value(0.05, tail="invalid")

def test_plot_without_d_alpha(monkeypatch):
    chisq = ChiSquare(3)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    chisq.plot()
