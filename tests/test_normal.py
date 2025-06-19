import pytest
from scipy.stats import norm
from hzero.distributions import Normal

def test_critic_value_one_tailed():
    normal = Normal(0,1)
    val = normal.critical_value(0.95, two_tailed=False)
    assert round(val, 4) == round(norm.ppf(0.95, 0, 1), 4)

def test_critic_value_two_tailed():
    normal = Normal(0,1)
    val = normal.critical_value(0.05, two_tailed=True)
    assert round(val, 4) == round(norm.ppf(0.05 / 2, 0, 1), 4)

def test_critic_value_invalid_alpha():
    normal = Normal(0,1)
    with pytest.raises(ValueError):
        normal.critical_value(1.5, True)

def test_p_value_two_tailed():
    normal = Normal(0,1)
    val = normal.p_value(1.5, tail="bilateral")
    expected = 2 * (1 - norm.cdf(1.5, 0, 1))
    assert round(val, 5) == round(expected, 5)

def test_p_value_left_tailed():
    normal = Normal(0,1)
    val = normal.p_value(1.5, tail="left")
    expected = norm.cdf(1.5, 0, 1)
    assert round(val, 5) == round(expected, 5)

def test_p_value_right_tailed():
    normal = Normal(0,1)
    val = normal.p_value(1.5, tail="right")
    expected = 1 - norm.cdf(1.5, 0, 1)
    assert round(val, 5) == round(expected, 5)

def test_plot_with_invalid_bilateral_tail():
    normal = Normal(0,1)
    with pytest.raises(ValueError):
        normal.plot(d=1.5, tail='bilateral')

def test_plot_valid(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    normal = Normal(0,1)
    normal.plot(d=1.5, alpha=0.05, tail='right')  # No debería lanzar errores


@pytest.mark.parametrize("alpha, two_tailed", [
    (0.10, False), (0.10, True),
    (0.01, False), (0.01, True),
    (0.001, False), (0.001, True)
])
def test_critic_value_varios_alphas(alpha, two_tailed):
    normal = Normal(0,1)
    val = normal.critical_value(alpha, two_tailed)
    expected = norm.ppf(alpha / 2, 0, 1) if two_tailed else norm.ppf(alpha, 0, 1)
    assert round(val, 4) == round(expected, 4)

@pytest.mark.parametrize("d", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_p_value_symmetry_two_tailed(d):
    normal = Normal(0,1)
    p1 = normal.p_value(d, tail='bilateral')
    p2 = normal.p_value(-d, tail='bilateral')
    assert round(p1, 6) == round(p2, 6)  # El p-valor bilateral debe ser simétrico

def test_plot_invalid_alpha():
    normal = Normal(0,1)
    with pytest.raises(ValueError):
        normal.plot(d=1.5, alpha=1.2, tail='right')

def test_plot_distribution_only(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    normal = Normal(0,1)
    normal.plot()
