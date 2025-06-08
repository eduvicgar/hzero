import pytest
from hzero.distributions import FSnedecor
from scipy.stats import f as fdistribution


def test_critic_value_right_tail():
    ftest = FSnedecor(10,12)
    result = ftest.critical_value(0.05, "right")
    expected = fdistribution.ppf(0.95, 10, 12)
    assert pytest.approx(result) == expected

def test_critic_value_left_tail():
    ftest = FSnedecor(10, 12)
    result = ftest.critical_value(0.05, "left")
    expected = fdistribution.ppf(0.05, 10, 12)
    assert pytest.approx(result) == expected

def test_critic_value_invalid_alpha_low():
    f = FSnedecor(10,12)
    with pytest.raises(ValueError):
        f.critical_value(-0.25, "left")

def test_critic_value_invalid_alpha_high():
    f = FSnedecor(10, 12)
    with pytest.raises(ValueError):
        f.critical_value(1.07, "left")

def test_p_value_right_tail():
    ftest = FSnedecor(10, 12)
    result = ftest.p_value(2.25, "right")
    expected = 1 - fdistribution.cdf(2.25, 10, 12)
    assert pytest.approx(result) == expected

def test_p_value_left_tail():
    ftest = FSnedecor(10, 12)
    result = ftest.p_value(0.97, "left")
    expected = fdistribution.cdf(0.97, 10, 12)
    assert pytest.approx(result) == expected

def test_p_value_zero():
    f = FSnedecor(10, 12)
    result = f.p_value(0.0, "right")
    assert 0 <= result <= 1

def test_p_value_negative_raises():
    f = FSnedecor(10, 12)
    with pytest.raises(ValueError):
        f.p_value(-0.25, "right")

def test_plot_right_tail(monkeypatch):
    f = FSnedecor(10, 12)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    f.plot(d=2.36, alpha=0.025, tail="right")

def test_plot_left_tail(monkeypatch):
    f = FSnedecor(10, 12)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    f.plot(d=2.36, alpha=0.025, tail="left")

def test_plot_bilateral(monkeypatch):
    f = FSnedecor(10, 12)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    f.plot(d=2.36, alpha=0.025, tail="bilateral")

def test_plot_missing_alpha_with_bilateral():
    f = FSnedecor(10, 12)
    with pytest.raises(ValueError):
        f.plot(d=2.36, tail="bilateral")

def test_plot_invalid_alpha_raises():
    f = FSnedecor(10, 12)
    with pytest.raises(ValueError):
        f.plot(d=2.36, alpha=1.03, tail="bilateral")

def test_plot_only_d(monkeypatch):
    f = FSnedecor(10, 12)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    f.plot(d=2.36, tail="right")

def test_init_df():
    f = FSnedecor(10, 12)
    assert f.df1 == 10 and f.df2 == 12

def test_multiple_calls_consistent():
    f = FSnedecor(10, 12)
    left = f.critical_value(0.01, "left")
    right = f.critical_value(0.01, "right")
    assert right > left

def test_invalid_tail_in_critic_value():
    f = FSnedecor(10, 12)
    with pytest.raises(ValueError):
        f.critical_value(0.02, "invalid")

def test_plot_without_d_alpha(monkeypatch):
    f = FSnedecor(10, 12)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    f.plot()
