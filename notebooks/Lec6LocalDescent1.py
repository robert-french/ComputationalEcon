import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="compact",                       # spans browser width
    css_file="marimo-overrides.css",       # hides marimo's floating outline
    app_title="Lecture 6: Local Descent I",  # title shown on the homepage card and notebook tab
)

@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from scipy.optimize import minimize


    return alt, minimize, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.sidebar(
        [
            mo.md('<a href="https://robert-french.github.io/" target="_self" style="display: block; margin-bottom: 1.5em;">Course home</a>'),
            mo.md("# [Lecture 6](#top)"),
            mo.md("**Local Descent in One Dimension**"),
            mo.nav_menu(
                {
                    "#sec1": "1. Intro to local descent",
                    "#sec2": "2. Direct search in 1D",
                    "#sec3": "3. Gradient descent in 1D",
                    "#sec4": "4. Step size & convergence",
                    "#sec5": "5. `scipy.optimize`",
                },
                orientation="vertical",
            ),
        ],
        width="260px",
    )
    return


@app.cell(hide_code=True)
def _():
    # Default x-axis range used by every plot in this notebook.
    # Edit these once if you want a wider/narrower default view across the board.
    PLOT_XMIN, PLOT_XMAX = -3.0, 3.0
    return PLOT_XMAX, PLOT_XMIN


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.md('<a href="https://robert-french.github.io/" target="_self">← Lecture 5</a>'),
            mo.md('<a href="https://robert-french.github.io/" target="_self">Lecture 7 →</a>'),
        ],
        justify="space-between", align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="top"></a>
    # Lecture 6: Local Descent in One Dimension
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contents

    1. [An introduction to local descent](#sec1)
    2. [Direct search in one dimension](#sec2)
    3. [Gradient descent in one dimension](#sec3)
    4. [Step size, gradient approximation, and convergence](#sec4)
    5. [The `scipy.optimize` package](#sec5)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sec1"></a>
    ## 1) An introduction to local descent
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we'll introduce another set of tools to solve optimization problems where it's difficult (or impossible) to obtain analytical solutions.

    A classic example from ECON 3100 is a firm choosing inputs to maximize profits. Suppose a firm chooses capital $k \ge 0$ and labor $l \ge 0$, produces output $q(k,l)$, sells output at price $p>0$, and pays input prices $r>0$ and $w>0$. We can write the firm's profit function as
    $$
    \pi(k,l) \;=\; p \;\cdot \, q(k,l)\;-\; r k \;-\; w l .
    $$
    The firm's problem is then
    $$
    \max_{k \ge 0,\; l \ge 0}\; \pi(k,l).
    $$
    In many settings, the production function $q(k,l)$ is complicated, so the best choice of $(k,l)$ may not be something we can solve for analytically. We showed in the past two lectures ways of solving these type of problems numerically. In this lecture we introduce **local descent algorithms** as another numerical optimization tool. A local descent algorithm is a simple idea:

    - Start with a guess for the choice variable(s);
    - Look at nearby alternatives;
    - Take a small step in the direction that increases (or decreases for minimization) the objective;
    - Repeat until the steps stop helping.

    A useful mental picture is walking downhill in fog. You can't see the whole landscape, but you can usually tell whether a small step forward, backward, left, or right goes downhill. If you keep taking downhill steps, you eventually reach a point where every small move seems to go uphill, so you stop. That stopping point is often a local minimum; the best point near where you started.

    In practice, there are two key questions at each step of a local descent algorithm.

    1. Which way is down?
    2. How big a step should one take?

    Most local descent algorithms use calculus to decide the downhill direction and a rule for how big a step to take. We'll begin considering these algorithms in one dimension where "nearby" just means "a little left" or "a little right", then extend the same logic to multiple dimensions next lecture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sec2"></a>
    ## 2) Direct search in one dimension
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The most basic kind of local descent algorithm is called **direct search**. Direct search algorithms don't use calculus to find the downhill direction. Instead these algorithms try a few simple candidate directions, evaluate the objective at those nearby points, and then move to whichever one improves the objective the most.

    In one dimension, this is almost trivial. From a current guess $x_t$, you can test a small step to the right and to the left, say $x_t+\Delta$ and $x_t-\Delta$. If one of them gives a better objective value, you move there; if neither improves things, you shrink the step size $\Delta$ and try again.

    These algorithms typically stop when either the size of the step, $\Delta$, becomes very small, *or* when the objective changes by a very small amount for a given step.

    The function below, `directSearchExample`,  implements a direct search in one dimension. In the function, the argument `f` is the objective function to be minimized (e.g., $f(x) = x^2$) and `x0` is the initial guess (e.g., $x = 10$). The keyword argument `Δ` sets the initial step size, `shrink` controls how much the step size is reduced when neither `x + Δ` nor `x - Δ` improves the objective, and `tol` determines when the step size is small enough to stop. The loop also stops if it reaches `max_iters`, which limits the maximum number of iterations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    def directSearchExample(f, x0, Δ=0.5, shrink=0.5, tol=1e-6, max_iters=10_000):

        x_best = float(x0)
        f_best = f(x_best)

        it = 0
        while (Δ >= tol) and (it < max_iters):

            # Try both directions
            x_plus  = x_best + Δ
            x_minus = x_best - Δ

            f_plus  = f(x_plus)
            f_minus = f(x_minus)

            # If either step improves, choose the best direction
            if (f_plus < f_best) or (f_minus < f_best):
                if f_plus <= f_minus:
                    x_best = x_plus
                    f_best = f_plus
                else:
                    x_best = x_minus
                    f_best = f_minus
            else:
                # If neither direction helped, shrink the step size
                Δ = shrink * Δ

            it += 1

        return x_best, f_best
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Visualize the direct search algorithm in one dimension**

    The interactive plot lets you visualize the steps that the direct search algorithm takes in one dimension. You can change the function that the plot displays using the dropdown beneath this text. You can also change the starting point, `x0`, and the step size, `Δ0`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    # Catalog of objective functions students can pick from.
    # Format: friendly label -> (LaTeX equation, callable f(x))
    func_options = {
        "Sine cubed":         (r"\sin(x - 2.05)^3 + 1.05",  lambda x: np.sin(x - 2.05) ** 3 + 1.05),
        "Parabola":           (r"(x - 1)^2",                 lambda x: (x - 1) ** 2),
        "Double well":        (r"(x^2 - 1)^2",               lambda x: (x ** 2 - 1) ** 2),
        "Quartic dip":        (r"x^4 - 3 x^2 + 2",           lambda x: x ** 4 - 3 * x ** 2 + 2),
        "Cosine + parabola":  (r"\cos(x) + x^2/10",          lambda x: np.cos(x) + x ** 2 / 10),
    }

    func_selector = mo.ui.dropdown(
        options=list(func_options.keys()),
        value="Sine cubed",
        label="Function to minimize",
    )
    return func_options, func_selector


@app.cell(hide_code=True)
def _(func_options, func_selector, mo):
    # Render the dropdown next to the LaTeX equation of the selected function.
    _eq_latex = func_options[func_selector.value][0]
    mo.hstack(
        [func_selector, mo.md(rf"$f(x) = {_eq_latex}$")],
        justify="start", align="center", gap=2,
    )
    return


@app.cell(hide_code=True)
def _(func_options, func_selector):
    # The chosen callable, exposed as `func` for the rest of the notebook.
    func = func_options[func_selector.value][1]
    return (func,)


@app.cell(hide_code=True)
def _(mo):
    x0_input = mo.ui.number(
        value=1.5, start=-10, stop=10, step=0.1, label=r"Starting point ($x_0$)",
    )
    Δ0_input = mo.ui.number(
        value=0.5, start=0.01, stop=5, step=0.05, label=r"Step size ($\Delta_0$)",
    )
    mo.hstack(
        [x0_input, Δ0_input],
        justify="start", align="center", gap=2,
    )
    return x0_input, Δ0_input


@app.cell(hide_code=True)
def _(x0_input, Δ0_input):
    x0 = x0_input.value
    Δ0 = Δ0_input.value
    max_steps = 5000
    shrink = 0.1
    tol = 1e-6
    return max_steps, shrink, tol, x0, Δ0


@app.function
def directSearchOneDim(f, x0, Δ0=0.5, shrink=0.5, tol=1e-6, max_steps=10_000):

    x = float(x0)
    Δ = float(Δ0)
    direction = 1

    xs = [x]
    fs = [f(x)]
    Δs = [Δ]
    dirs = [direction]

    t = 0
    while (Δ >= tol) and (t < max_steps):

        fx = fs[-1]

        # try current direction
        x_try = x + direction * Δ
        f_try = f(x_try)

        if f_try <= fx:
            x = x_try
            dirs.append(direction)
        else:
            # reverse once and try
            direction = -direction
            x_try2 = x + direction * Δ
            f_try2 = f(x_try2)

            if f_try2 <= fx:
                x = x_try2
                dirs.append(direction)
            else:
                # neither helped: shrink and stay put
                Δ *= shrink
                dirs.append(direction)

        f_new = f(x)
        xs.append(x)
        fs.append(f_new)
        Δs.append(Δ)

        t += 1

    return xs, fs, Δs, dirs


@app.cell(hide_code=True)
def _(func, max_steps, shrink, tol, x0, Δ0):
    xs, fs, Δs, dirs = directSearchOneDim(
        func, x0, Δ0=Δ0, shrink=shrink, tol=tol, max_steps=max_steps
    )
    return fs, xs, Δs


@app.cell(hide_code=True)
def _(mo, xs):
    step_ds_slider = mo.ui.slider(
        start=1, stop=len(xs), value=1, step=1,
        label="Step k", show_value=True,
    )
    step_ds_slider
    return (step_ds_slider,)


@app.cell(hide_code=True)
def _(PLOT_XMAX, PLOT_XMIN, alt, fs, func, mo, np, pd, step_ds_slider, xs, Δs):
    # Fixed default x-range so plots are comparable across functions and sections.
    _xmin, _xmax = PLOT_XMIN, PLOT_XMAX
    _xgrid = np.linspace(_xmin, _xmax, 500)

    _kk = max(1, min(step_ds_slider.value, len(xs)))

    _curve_y = [func(x) for x in _xgrid]

    # Place the converged minimum 1/3 from the bottom of the plot
    # (so curve has 2x more room above the minimum than below).
    _y_center = fs[-1]
    _max_above = max(
        max((_y - _y_center for _y in _curve_y), default=0.1),
        max((_y - _y_center for _y in fs), default=0.1),
        0.1,
    )
    _y_above = _max_above * 1.05
    _y_below = _y_above / 2
    _ymin = _y_center - _y_below
    _ymax = _y_center + _y_above
    _y_scale = alt.Scale(domain=[_ymin, _ymax])

    _curve_df = pd.DataFrame({"x": _xgrid, "y": _curve_y})
    _past_df = pd.DataFrame({
        "x": xs[:_kk - 1], "y": fs[:_kk - 1],
        "step": list(range(1, _kk)),
    })
    _curr_df = pd.DataFrame({
        "x": [xs[_kk - 1]], "y": [fs[_kk - 1]], "step": [_kk],
    })

    _tooltip = [
        alt.Tooltip("step:Q"),
        alt.Tooltip("x:Q", format=".4f"),
        alt.Tooltip("y:Q", format=".4f"),
    ]

    _curve = alt.Chart(_curve_df).mark_line(
        strokeWidth=3, color="#4C72B0",
    ).encode(
        x=alt.X("x:Q", title="𝑥"),
        y=alt.Y("y:Q", title="𝑓(𝑥)", scale=_y_scale),
    )

    _past = alt.Chart(_past_df).mark_circle(
        size=90, opacity=0.55, color="#4C72B0",
    ).encode(x="x:Q", y=alt.Y("y:Q", scale=_y_scale), tooltip=_tooltip)

    _curr = alt.Chart(_curr_df).mark_point(
        size=220, color="crimson", filled=True,
        stroke="white", strokeWidth=2,
    ).encode(x="x:Q", y=alt.Y("y:Q", scale=_y_scale), tooltip=_tooltip)

    _title = (
        rf"$\text{{Step }} k = {_kk}, "
        rf"\quad x_k = {xs[_kk - 1]:.4f}, "
        rf"\quad f(x_k) = {fs[_kk - 1]:.4f}, "
        rf"\quad \Delta_k = {Δs[_kk - 1]:.4f}$"
    )

    _chart = (_curve + _past + _curr).properties(
        width=620, height=340,
    ).configure_view(stroke=None).configure_axis(
        gridColor="#eaeaea", domainColor="#888", labelFontSize=11,
        titleFontSize=12,
    ).interactive()
    mo.vstack([mo.md(_title), _chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Try it yourself!**

        Experiment with the visualization. Choose a different function, choose different initial settings, and move the slider to understand how direct search works and responds to its settings.
        """),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sec3"></a>
    ## 3) Gradient descent in one dimension
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The most basic kind of local descent algorithm that *does* use calculus is **gradient descent**. Gradient descent uses the derivative of the objective function to decide which way is downhill. Intuitively, if the slope at the current point is positive, then moving left decreases the function; if the slope is negative, then moving right decreases the function.

    In one dimension, gradient descent uses a very simple rule. Starting from a current guess $x_t$, compute the derivative $f'(x_t)$ and take a step in the opposite direction to reduce the objective:
    $$
    x_{t+1} \;=\; x_t \;-\;\alpha\, f'(x_t),
    $$
    where $\alpha>0$ is the step size (also called the learning rate). If $\alpha$ is too large, the algorithm can overshoot the minimum and bounce around; if $\alpha$ is too small, progress can be very slow.

    These algorithms typically stop when either $f'(x_t)$ becomes very small (meaning the function is nearly flat at the current point), or when the change in the objective between iterations becomes very small.

    The function below, `gradientDescentOneDim`, implements gradient descent in one dimension. In the function, the argument `f` is the objective function to be minimized (e.g., $f(x)=x^2$), `h` helps us approximate its derivative (e.g., $f'(x)=2x$), and `x0` is the initial guess. The keyword argument `α` sets the step size, `tol` determines the stopping threshold (typically applied to $f'(x_t)$ or the improvement in $f$), and `max_iters` limits the maximum number of iterations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    def gradientDescentOneDim(f, x0, α=0.1, h=1e-6, tol=1e-6, max_iters=10_000):

        x_best = float(x0)
        f_best = f(x_best)

        # Central-difference derivative approximation
        def df(x):
            return (f(x + h) - f(x - h)) / (2 * h)

        it = 0
        while (abs(df(x_best)) >= tol) and (it < max_iters):
            x_best = x_best - α * df(x_best)
            f_best = f(x_best)
            it += 1

        return x_best, f_best
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Visualize the gradient descent algorithm in one dimension**

    Just like with the direct search plot, this interactive plot lets you visualize the steps that the gradient descent algorithm takes in one dimension. Change the function that the plot displays using the dropdown beneath this text. You can also change the starting point, `x0`, and the step size, `α`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    # Catalog of objective functions for the gradient-descent demo.
    gd_func_options = {
        "Sine cubed":         (r"\sin(x - 2.05)^3 + 1.05",  lambda x: np.sin(x - 2.05) ** 3 + 1.05),
        "Parabola":           (r"(x - 1)^2",                 lambda x: (x - 1) ** 2),
        "Double well":        (r"(x^2 - 1)^2",               lambda x: (x ** 2 - 1) ** 2),
        "Quartic dip":        (r"x^4 - 3 x^2 + 2",           lambda x: x ** 4 - 3 * x ** 2 + 2),
        "Cosine + parabola":  (r"\cos(x) + x^2/10",          lambda x: np.cos(x) + x ** 2 / 10),
    }
    gd_func_selector = mo.ui.dropdown(
        options=list(gd_func_options.keys()),
        value="Sine cubed",
        label="Function to minimize",
    )
    return gd_func_options, gd_func_selector


@app.cell(hide_code=True)
def _(gd_func_options, gd_func_selector, mo):
    _eq_latex = gd_func_options[gd_func_selector.value][0]
    mo.hstack(
        [gd_func_selector, mo.md(rf"$f(x) = {_eq_latex}$")],
        justify="start", align="center", gap=2,
    )
    return


@app.cell(hide_code=True)
def _(gd_func_options, gd_func_selector):
    localDescentObj = gd_func_options[gd_func_selector.value][1]
    return (localDescentObj,)


@app.cell(hide_code=True)
def _(mo):
    gd_x0_input = mo.ui.number(
        value=2.0, start=-10, stop=10, step=0.1, label=r"Starting point ($x_0$)",
    )
    gd_α_input = mo.ui.number(
        value=0.01, start=0.0001, stop=1, step=0.005, label=r"Step size ($\alpha$)",
    )
    mo.hstack(
        [gd_x0_input, gd_α_input],
        justify="start", align="center", gap=2,
    )
    return gd_x0_input, gd_α_input


@app.cell(hide_code=True)
def _(gd_x0_input, gd_α_input):
    gd_x0 = gd_x0_input.value
    gd_α = gd_α_input.value
    gd_max_steps = 100
    gd_tol = 1e-6
    return gd_max_steps, gd_tol, gd_x0, gd_α


@app.function
def gradDescOneDim(obj, x_start, η=0.1, h=1e-6, tol=1e-6, max_iters=10_000):
    # Central-difference derivative approximation
    def obj_prime(z):
        return (obj(z + h) - obj(z - h)) / (2 * h)

    gd_x_path = [float(x_start)]
    gd_f_path = [obj(gd_x_path[0])]

    t = 0
    while (abs(obj_prime(gd_x_path[-1])) >= tol) and (t < max_iters):
        x_now = gd_x_path[-1]
        x_next = x_now - η * obj_prime(x_now)
        gd_x_path.append(x_next)
        gd_f_path.append(obj(x_next))
        t += 1

    return gd_x_path, gd_f_path


@app.cell(hide_code=True)
def _(gd_max_steps, gd_tol, gd_x0, gd_α, localDescentObj):
    gd_xs, gd_fs = gradDescOneDim(
        localDescentObj, gd_x0, η=gd_α, tol=gd_tol, max_iters=gd_max_steps
    )
    return (gd_xs,)


@app.cell(hide_code=True)
def _(gd_xs, mo):
    step_gd_slider = mo.ui.slider(
        start=1, stop=len(gd_xs), value=1, step=1,
        label="Step k", show_value=True,
    )
    step_gd_slider
    return (step_gd_slider,)


@app.cell(hide_code=True)
def _(PLOT_XMAX, PLOT_XMIN, alt, gd_xs, localDescentObj, mo, np, pd, step_gd_slider):
    # Derivative for tangent visualization (central difference)
    _h_tan = 1e-6
    def _gd_obj_prime(z):
        return (localDescentObj(z + _h_tan) - localDescentObj(z - _h_tan)) / (2 * _h_tan)

    # Fixed default x-range so plots are comparable across functions and sections.
    _xmin, _xmax = PLOT_XMIN, PLOT_XMAX
    _xgrid = np.linspace(_xmin, _xmax, 600)

    _kk = max(1, min(step_gd_slider.value, len(gd_xs)))
    _xk = gd_xs[_kk - 1]
    _fk = localDescentObj(_xk)
    _m = _gd_obj_prime(_xk)

    # Place the converged minimum 1/3 from the bottom of the plot
    # (so curve has 2x more room above the minimum than below).
    _curve_y = [localDescentObj(x) for x in _xgrid]
    _path_y = [localDescentObj(x) for x in gd_xs]
    _y_center = localDescentObj(gd_xs[-1])
    _max_above = max(
        max((_y - _y_center for _y in _curve_y), default=0.1),
        max((_y - _y_center for _y in _path_y), default=0.1),
        0.1,
    )
    _y_above = _max_above * 1.05
    _y_below = _y_above / 2
    _ymin = _y_center - _y_below
    _ymax = _y_center + _y_above

    _tan_y = _fk + _m * (_xgrid - _xk)

    _obj_df = pd.DataFrame({
        "x": _xgrid,
        "y": [localDescentObj(x) for x in _xgrid],
    })
    _tan_df = pd.DataFrame({"x": _xgrid, "y": _tan_y})
    _past_df = pd.DataFrame({
        "x": gd_xs[:_kk - 1],
        "y": [localDescentObj(x) for x in gd_xs[:_kk - 1]],
        "step": list(range(1, _kk)),
    })
    _curr_df = pd.DataFrame({
        "x": [_xk], "y": [_fk], "step": [_kk],
    })

    _y_scale = alt.Scale(domain=[_ymin, _ymax])
    _tooltip = [
        alt.Tooltip("step:Q"),
        alt.Tooltip("x:Q", format=".4f"),
        alt.Tooltip("y:Q", format=".4f"),
    ]

    _obj = alt.Chart(_obj_df).mark_line(
        strokeWidth=3, color="#4C72B0",
    ).encode(
        x=alt.X("x:Q", title="𝑥"),
        y=alt.Y("y:Q", title="𝑓(𝑥)", scale=_y_scale),
    )

    _tan = alt.Chart(_tan_df).mark_line(
        color="royalblue", strokeDash=[6, 4],
        opacity=0.55, strokeWidth=2,
    ).encode(
        x="x:Q",
        y=alt.Y("y:Q", scale=_y_scale),
    )

    _past = alt.Chart(_past_df).mark_circle(
        size=90, opacity=0.55, color="#4C72B0",
    ).encode(
        x="x:Q",
        y=alt.Y("y:Q", scale=_y_scale),
        tooltip=_tooltip,
    )

    _curr = alt.Chart(_curr_df).mark_point(
        size=220, color="crimson", filled=True,
        stroke="white", strokeWidth=2,
    ).encode(
        x="x:Q",
        y=alt.Y("y:Q", scale=_y_scale),
        tooltip=_tooltip,
    )

    _title = (
        rf"$\text{{Step }} k = {_kk}, "
        rf"\quad x_k = {_xk:.3f}, "
        rf"\quad f(x_k) = {_fk:.3f}, "
        rf"\quad f'(x_k) = {_m:.3f}$"
    )

    _chart = (_obj + _tan + _past + _curr).properties(
        width=620, height=340,
    ).configure_view(stroke=None).configure_axis(
        gridColor="#eaeaea", domainColor="#888",
        labelFontSize=11, titleFontSize=12,
    ).interactive()
    mo.vstack([mo.md(_title), _chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Try it yourself!**

        Experiment with the visualization. Choose a different function, choose different initial settings, and move the slider to understand how gradient descent works in one dimension.
        """),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sec4"></a>
    ## 4) Step size, gradient approximation, and convergence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Step size**

    Step size matters because it determines how aggressively a local descent algorithm moves each time it updates its guess. In an update like $x_{t+1} = x_t - \alpha f'(x_t)$, the derivative $f'(x_t)$ tells you which way to move, but the *step size* $\alpha$ tells you how far to move. If $\alpha$ is too large, the algorithm can jump past the minimum and bounce back and forth, or even diverge (the objective gets worse rather than better). If $\alpha$ is too small, the algorithm will usually move in the right direction but can take a very long time to make progress. Many practical implementations therefore choose $\alpha$ using a simple rule (a fixed small step size), or by using what's called a *line search*, which tries a few candidate step sizes and picks one that reduces the objective enough without taking an unstable jump.

    **Gradient approximation**

    Gradient approximation is about how the algorithm decides what direction counts as downhill. In gradient descent, the natural direction is the negative gradient. In one dimension this is $-f'(x_t)$, and in two dimensions it is denoted by $-\nabla f(x_t, y_t)$. Computing that derivative can be done in a few ways. You can derive it by hand for simple functions, approximate it numerically using what is called *finite differences* (e.g., comparing $f(x+h)$ and $f(x-h)$ for a small $h$), or compute it using *automatic differentiation*, wherein Python will compute the derivatives analytically based on the functions you have written. Different methods trade off simplicity, speed, and numerical accuracy, but they all serve the same purpose of finding a direction that points downhill.

    **Convergence**

    Convergence refers to whether the algorithm actually settles down near an optimum. A common convergence check is that the gradient becomes small, such as $|f'(x_t)| <$ `tol`, which signals that the function is nearly flat at the current point (a necessary condition for an interior minimum/maximum). Another check is that the objective stops changing much, such as $|f(x_{t+1}) - f(x_t)|$ becoming tiny, or that the step sizes become very small. Importantly, though, these are local algorithms; they are designed to find a nearby optimum based on the starting guess, and they can converge to different minima/maxima if the function has multiple valleys/hills.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sec5"></a>
    ## 5) The `scipy.optimize` package
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far, we've implemented some simple local descent algorithms ourselves. In practice, you will often use a *library optimizer* that has good defaults and lots of tested methods.

    In Python, a common choice is the package `scipy.optimize`. A package is a reusable bundle of code written by others that you can add to your project and load by typing in a cell `from scipy.optimize import minimize`.

    `scipy.optimize` provides ready-to-run implementations of many standard optimization algorithms, such as gradient descent, and other similar algorithms we haven't covered including conjugate gradient and quasi-Newton methods. When using packages like `scipy.optimize` you can focus on defining an objective function rather than coding an algorithm from scratch. In practice, you typically call `minimize` with your objective (and sometimes its derivative), choose an algorithm, and then read the solution and diagnostics from the returned result object to see what it found and how it converged.

    We won't cover other local descent algorithms in detail as they all follow a similar structure. Instead, here we outline some key features of common algorithms that you may encounter through packages like `scipy.optimize`. Below this Markdown cell, you can select any one of these algorithms and observe in the visualization how they converge (or not) to a local minimum given your initial settings and your specified function. We will get more practice implementing local descent algorithms in `scipy.optimize` next lecture with multidimensional optimization problems, where the package really shines.

    - **Gradient Descent (GD)**: Moves "downhill" by using the slope at the current point, taking a step like $x_{t+1}=x_t-\alpha f'(x_t)$. In practice, a *line search* often chooses the step size $\alpha$ by trying a few candidate step sizes and picking one that decreases the objective.

    - **Conjugate Gradient (CG)**: It uses functions' slopes, but it chooses each new direction in a way that avoids "undoing" progress from earlier directions. You can think of it as a smarter version of gradient descent that reduces any back-and-forth zig-zagging. It is often combined with a line search for choosing the step size.

    - **Broyden–Fletcher–Goldfarb–Shanno (BFGS)**: It uses slope information from recent steps to learn the shape of the objective near the current point. That lets it choose directions and step sizes more intelligently (typically with a line search), so it often reaches a solution in fewer steps than gradient descent.

    - **Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS)**: Like BFGS, but it only remembers a small amount of recent slope information rather than storing lots of past information. This makes it faster and more practical when there are many choice variables, while still usually converging in fewer steps than basic gradient descent.

    - **Nelder–Mead (NelderMead)**: It does not use slopes at all. It keeps a small "cloud" of trial points and repeatedly shifts and reshapes that cloud by comparing objective values, moving it toward lower values. Because it is derivative-free, it is useful when derivaties are hard to compute, but it can be slower or less reliable on some problems.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    # Catalog of objective functions for the scipy.optimize demo.
    alg_func_options = {
        "Sine cubed":         (r"\sin(x - 2.05)^3 + 1.05",  lambda x: np.sin(x - 2.05) ** 3 + 1.05),
        "Parabola":           (r"(x - 1)^2",                 lambda x: (x - 1) ** 2),
        "Double well":        (r"(x^2 - 1)^2",               lambda x: (x ** 2 - 1) ** 2),
        "Quartic dip":        (r"x^4 - 3 x^2 + 2",           lambda x: x ** 4 - 3 * x ** 2 + 2),
        "Cosine + parabola":  (r"\cos(x) + x^2/10",          lambda x: np.cos(x) + x ** 2 / 10),
    }
    alg_func_selector = mo.ui.dropdown(
        options=list(alg_func_options.keys()),
        value="Sine cubed",
        label="Function to minimize",
    )
    return alg_func_options, alg_func_selector


@app.cell(hide_code=True)
def _(alg_func_options, alg_func_selector, mo):
    _eq_latex = alg_func_options[alg_func_selector.value][0]
    mo.hstack(
        [alg_func_selector, mo.md(rf"$f(x) = {_eq_latex}$")],
        justify="start", align="center", gap=2,
    )
    return


@app.cell(hide_code=True)
def _(alg_func_options, alg_func_selector):
    gdplot_obj = alg_func_options[alg_func_selector.value][1]

    def gdplot_objv(z):
        return gdplot_obj(z[0])

    return gdplot_obj, gdplot_objv


@app.cell(hide_code=True)
def _(mo):
    alg_x0_input = mo.ui.number(
        value=1.5, start=-10, stop=10, step=0.1, label=r"Starting point ($x_0$)",
    )
    return (alg_x0_input,)


@app.cell(hide_code=True)
def _(alg_x0_input):
    alg_x0 = alg_x0_input.value
    alg_maxit = 100
    alg_gtol = 0.01
    return alg_gtol, alg_maxit, alg_x0


@app.function
def gdplot_make_logger_1d(objv):
    x_eval = []
    f_eval = []

    def obj_logged(z):
        x = z[0]
        fx = objv(z)
        x_eval.append(float(x))
        f_eval.append(float(fx))
        return fx

    return obj_logged, x_eval, f_eval


@app.cell(hide_code=True)
def _(alg_gtol, alg_maxit, alg_x0, gdplot_objv, minimize, np):
    gdplot_z0 = np.array([alg_x0], dtype=float)

    def run_gd(objv_logged, objv_raw, z0, α, gtol, max_iters):
        x = float(z0[0])
        h = 1e-6
        for _ in range(max_iters):
            objv_logged(np.array([x]))
            gp = objv_raw(np.array([x + h]))
            gm = objv_raw(np.array([x - h]))
            grad = (gp - gm) / (2 * h)
            if abs(grad) < gtol:
                break
            x = x - α * grad
        return x

    gdplot_methods = ["GD", "CG", "BFGS", "LBFGS", "NelderMead"]

    gdplot_logs = {}

    for name in gdplot_methods:
        obj_logged, x_eval, f_eval = gdplot_make_logger_1d(gdplot_objv)
        if name == "GD":
            run_gd(obj_logged, gdplot_objv, gdplot_z0, α=0.1,
                   gtol=alg_gtol, max_iters=alg_maxit)
            res = None
        elif name == "CG":
            res = minimize(obj_logged, gdplot_z0, method="CG",
                           options={"gtol": alg_gtol, "maxiter": alg_maxit})
        elif name == "BFGS":
            res = minimize(obj_logged, gdplot_z0, method="BFGS",
                           options={"gtol": alg_gtol, "maxiter": alg_maxit})
        elif name == "LBFGS":
            res = minimize(obj_logged, gdplot_z0, method="L-BFGS-B",
                           options={"gtol": alg_gtol, "maxiter": alg_maxit})
        elif name == "NelderMead":
            res = minimize(obj_logged, gdplot_z0, method="Nelder-Mead",
                           options={"xatol": alg_gtol, "fatol": alg_gtol,
                                    "maxiter": alg_maxit})
        gdplot_logs[name] = (x_eval, f_eval, res)
    return (gdplot_logs,)


@app.cell(hide_code=True)
def _(mo):
    alg_method_dropdown = mo.ui.dropdown(
        options=["GD", "CG", "BFGS", "LBFGS", "NelderMead"],
        value="GD",
        label="Algorithm",
    )
    return (alg_method_dropdown,)


@app.cell(hide_code=True)
def _(alg_method_dropdown, alg_x0_input, mo):
    mo.hstack(
        [alg_method_dropdown, alg_x0_input],
        justify="start", align="center", gap=2,
    )
    return


@app.cell(hide_code=True)
def _(alg_method_dropdown, gdplot_logs):
    # Slider — which *evaluation* to show (depends on chosen algorithm)
    x_eval_sel, f_eval_sel, _res_sel = gdplot_logs[alg_method_dropdown.value]

    # de-duplicate consecutive repeats (keeps the visualization cleaner)
    gdplot_xs_sel = []
    gdplot_fs_sel = []
    for i in range(len(x_eval_sel)):
        if i == 0 or x_eval_sel[i] != x_eval_sel[i - 1]:
            gdplot_xs_sel.append(x_eval_sel[i])
            gdplot_fs_sel.append(f_eval_sel[i])
    return gdplot_fs_sel, gdplot_xs_sel


@app.cell(hide_code=True)
def _(gdplot_xs_sel, mo):
    eval_k_slider = mo.ui.slider(
        start=1, stop=max(1, len(gdplot_xs_sel)), value=1, step=1,
        label="Eval k", show_value=True,
    )
    eval_k_slider
    return (eval_k_slider,)


@app.cell(hide_code=True)
def _(
    PLOT_XMAX,
    PLOT_XMIN,
    alg_method_dropdown,
    alg_x0,
    alt,
    eval_k_slider,
    gdplot_fs_sel,
    gdplot_obj,
    gdplot_xs_sel,
    mo,
    np,
    pd,
):
    if len(gdplot_xs_sel) == 0:
        _chart = alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(
            text="No evaluations recorded.", fontSize=16, color="#666",
        ).encode(x="x:Q", y="y:Q").properties(width=620, height=340)
    else:
        _kk = max(1, min(eval_k_slider.value, len(gdplot_xs_sel)))

        # Fixed default x-range so plots are comparable across functions and sections.
        _xmin, _xmax = PLOT_XMIN, PLOT_XMAX
        _xgrid = np.linspace(_xmin, _xmax, 700)

        _curve_y = [gdplot_obj(x) for x in _xgrid]

        # Place the converged minimum 1/3 from the bottom of the plot
        # (so curve has 2x more room above the minimum than below).
        _y_center = gdplot_fs_sel[-1]
        _max_above = max(
            max((_y - _y_center for _y in _curve_y), default=0.1),
            max((_y - _y_center for _y in gdplot_fs_sel), default=0.1),
            0.1,
        )
        _y_above = _max_above * 1.05
        _y_below = _y_above / 2
        _ymin = _y_center - _y_below
        _ymax = _y_center + _y_above
        _y_scale = alt.Scale(domain=[_ymin, _ymax])

        _obj_df = pd.DataFrame({"x": _xgrid, "y": _curve_y})
        _past_df = pd.DataFrame({
            "x": gdplot_xs_sel[:_kk - 1],
            "y": gdplot_fs_sel[:_kk - 1],
            "eval": list(range(1, _kk)),
        })
        _curr_df = pd.DataFrame({
            "x": [gdplot_xs_sel[_kk - 1]],
            "y": [gdplot_fs_sel[_kk - 1]],
            "eval": [_kk],
        })

        _tooltip = [
            alt.Tooltip("eval:Q"),
            alt.Tooltip("x:Q", format=".4f"),
            alt.Tooltip("y:Q", format=".4f"),
        ]

        _obj = alt.Chart(_obj_df).mark_line(
            strokeWidth=3, color="#4C72B0",
        ).encode(
            x=alt.X("x:Q", title="𝑥"),
            y=alt.Y("y:Q", title="𝑓(𝑥)", scale=_y_scale),
        )

        _past = alt.Chart(_past_df).mark_circle(
            size=90, opacity=0.55, color="#4C72B0",
        ).encode(x="x:Q", y=alt.Y("y:Q", scale=_y_scale), tooltip=_tooltip)

        _curr = alt.Chart(_curr_df).mark_point(
            size=220, color="crimson", filled=True,
            stroke="white", strokeWidth=2,
        ).encode(x="x:Q", y=alt.Y("y:Q", scale=_y_scale), tooltip=_tooltip)

        _title = (
            rf"$\text{{Method}} = \text{{{alg_method_dropdown.value}}}, "
            rf"\quad \text{{Eval step}} = {_kk}, "
            rf"\quad x_0 = {alg_x0:.2f}$"
        )

        _chart = mo.vstack([
            mo.md(_title),
            (_obj + _past + _curr).properties(
                width=620, height=340,
            ).configure_view(stroke=None).configure_axis(
                gridColor="#eaeaea", domainColor="#888",
                labelFontSize=11, titleFontSize=12,
            ).interactive(),
        ])

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Try it yourself!**

        Experiment with the different algorithm options. Are any of the algorithms more or less sensitive to the initial values or function form?
        """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
