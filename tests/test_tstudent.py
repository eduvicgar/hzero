import pytest
from scipy.stats import t as tdistribution
from hzero.distributions import TStudent

def test_critic_value_one_tailed():
    ttest = TStudent(10)
    val = ttest.critical_value(0.05, two_tailed=False)
    assert round(val, 4) == round(tdistribution.ppf(0.95, 10), 4)

def test_critic_value_two_tailed():
    ttest = TStudent(10)
    val = ttest.critical_value(0.05, two_tailed=True)
    assert round(val, 4) == round(tdistribution.ppf(0.975, 10), 4)

def test_critic_value_invalid_alpha():
    t = TStudent(10)
    with pytest.raises(ValueError):
        t.critical_value(1.5, True)

def test_p_value_two_tailed():
    ttest = TStudent(10)
    val = ttest.p_value(1.5, two_tailed=True)
    expected = 2 * (1 - tdistribution.cdf(1.5, 10))
    assert round(val, 5) == round(expected, 5)

def test_p_value_one_tailed():
    ttest = TStudent(10)
    val = ttest.p_value(1.5, two_tailed=False)
    expected = tdistribution.cdf(1.5, 10)
    assert round(val, 5) == round(expected, 5)

def test_plot_with_invalid_bilateral_tail():
    t = TStudent(10)
    with pytest.raises(ValueError):
        t.plot(d=1.5, tail='bilateral')

def test_plot_valid(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    t = TStudent(10)
    t.plot(d=1.5, alpha=0.05, tail='right')  # No debería lanzar errores


@pytest.mark.parametrize("alpha, two_tailed", [
    (0.10, False), (0.10, True),
    (0.01, False), (0.01, True),
    (0.001, False), (0.001, True)
])
def test_critic_value_varios_alphas(alpha, two_tailed):
    ttest = TStudent(20)
    val = ttest.critical_value(alpha, two_tailed)
    expected = tdistribution.ppf(1 - alpha / 2, 20) if two_tailed else tdistribution.ppf(1 - alpha, 20)
    assert round(val, 4) == round(expected, 4)

@pytest.mark.parametrize("d", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_p_value_symmetry_two_tailed(d):
    t = TStudent(12)
    p1 = t.p_value(d, two_tailed=True)
    p2 = t.p_value(-d, two_tailed=True)
    assert round(p1, 6) == round(p2, 6)  # La p-value bilateral debe ser simétrica

def test_plot_invalid_alpha():
    t = TStudent(12)
    with pytest.raises(ValueError):
        t.plot(d=1.5, alpha=1.2, tail='right')

def test_plot_distribution_only(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    t = TStudent(10)
    t.plot()
