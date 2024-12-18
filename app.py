from shiny import App, ui, render, reactive
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------- UI -------------------
app_ui = ui.page_fluid(
    ui.h2("SIR Model for Epidemic Spread"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("preset", "Select Preset Values:", {
                "none": "Custom Values",
                "preset1": "S0=997, I0=3, R0=0, Beta=0.4, Gamma=0.04",
                "preset2": "S0=950, I0=50, R0=0, Beta=0.3, Gamma=0.1",
                "preset3": "S0=900, I0=100, R0=0, Beta=0.2, Gamma=0.05"
            }),
            ui.input_slider("beta", "Infection Rate (Beta):", min=0.0, max=1.0, value=0.3, step=0.01),
            ui.input_slider("gamma", "Recovery Rate (Gamma):", min=0.0, max=1.0, value=0.1, step=0.01),
            ui.input_numeric("s0", "Initial Susceptible (S0):", value=990, min=0),
            ui.input_numeric("i0", "Initial Infected (I0):", value=10, min=0),
            ui.input_numeric("r0", "Initial Recovered (R0):", value=0, min=0),
            ui.input_slider("t_max", "Time (t_max):", min=10, max=365, value=100),
            ui.input_action_button("simulate", "Simulate")
        ),
        ui.output_plot("sir_plot")
    )
)


# ---------------- Server -------------------
def server(input, output, session):
    def sir_model(t, y, beta, gamma):
        S, I, R = y
        N = S + I + R  # Total population remains constant
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    @output
    @render.plot
    @reactive.event(input.simulate)
    def sir_plot():
        # Check the selected preset and apply values
        preset = input.preset()
        if preset == "preset1":
            S0, I0, R0, beta, gamma = 997, 3, 0, 0.4, 0.04
        elif preset == "preset2":
            S0, I0, R0, beta, gamma = 950, 50, 0, 0.3, 0.1
        elif preset == "preset3":
            S0, I0, R0, beta, gamma = 900, 100, 0, 0.2, 0.05
        else:
            S0 = input.s0()
            I0 = input.i0()
            R0 = input.r0()
            beta = input.beta()
            gamma = input.gamma()

        t_max = input.t_max()

        # Initial conditions
        y0 = [S0, I0, R0]
        t_eval = np.linspace(0, t_max, 1000)

        # Solve the system of ODEs
        solution = solve_ivp(sir_model, [0, t_max], y0, args=(beta, gamma), t_eval=t_eval)
        S, I, R = solution.y

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(solution.t, S, label='Susceptible', color='blue')
        plt.plot(solution.t, I, label='Infected', color='red')
        plt.plot(solution.t, R, label='Recovered', color='green')

        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('SIR Model Dynamics')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


# ---------------- Run App -------------------
app = App(app_ui, server)
