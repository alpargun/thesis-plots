make sure pdflatex is installed on your system.

Save plots in PGF format.
PGF allows for scalable figures as it is in vector form.


To include in LaTeX:

\begin{figure}
    \begin{center}
        \input{histogram.pgf}
    \end{center}
    \caption{A PGF histogram from \texttt{matplotlib}.}
\end{figure}


Adjust plot size:

\usepackage{layouts}

[...]

\printinunitsof{in}\prntlen{\textwidth}

then adjust the matplotlib figure accordingly:

fig.set_size_inches(w=4.7747, h=3.5)


Try \usepackage{pgfplots} or \usepackage{pgf}


Reference: https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
