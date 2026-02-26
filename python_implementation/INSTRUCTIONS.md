# Instructions for Running the C++ Implementation

This document provides a step-by-step guide to compiling and running the optimized C++ implementation of the $sl_3$ Hecke algebra solver on your machine (macOS with Intel chip).

## Prerequisites

You need a C++ compiler (like `clang++` or `g++`) and `make` installed. These are typically available on macOS if you have installed **Xcode Command Line Tools**.

 To check if you have them, open your **Terminal** and run:
```bash
g++ --version
make --version
```
If you don't have them, you can install them by running:
```bash
xcode-select --install
```

## Quick Start

1.  **Navigate to the C++ directory**
    Open your terminal and change directory (`cd`) to the folder containing the C++ code:
    ```bash
    cd cpp_implementation
    ```

2.  **Compile the code**
    Run the `make` command. This uses the instructions in the `Makefile` to build the program.
    ```bash
    make
    ```
    *   If successful, you will see output lines starting with `g++ ...`.
    *   It will produce an executable file named `sanity_check`.

3.  **Run the Sanity Check**
    Execute the program by running:
    ```bash
    ./sanity_check
    ```
    This will print the basis vectors, the action of generator $e_1$, and the explicit matrices for $H$ and $T$ for a small system size ($L=2$).

## Modifying the Code

The current `sanity_check.cpp` is set up for a small system ($L=2$) to verify correctness. To run larger simulations or change parameters:

1.  **Open `sanity_check.cpp`** (or create a new `.cpp` file based on it) in a text editor.
2.  **Change Parameters:**
    Look for the lines:
    ```cpp
    int L = 2;
    Complex n_val = 1.0;
    ```
    Change `L` to your desired size (e.g., `3`, `4`) and `n_val` to your desired parameter value.
3.  **Recompile:**
    Every time you change the C++ code (`.cpp` or `.h` files), you must run `make` again to update the executable.

## What to Keep an Eye On

*   **Compilation Warnings:** You might see warnings like `comparison of integer expressions`. These are generally harmless in this context (comparing signed vs unsigned integers in loops), but ensure there are no **Errors** (which stop compilation).
*   **Performance:** For larger $L$ (e.g., $L=5$ or $6$), the dimension grows rapidly. The computation might take significantly longer.
*   **Output Size:** Printing explicit matrices for $L > 3$ produces very large output. It is better to only print eigenvalues or specific statistics for larger systems.

## Cleaning Up

To remove the compiled files and start fresh, run:
```bash
make clean
```
