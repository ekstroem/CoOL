# Causes of Outcome Learning

To install you can run the following code in R

```{r}
# Make sure the devtools package is installed using: if(!require(“devtools”)) install.packages(“devtools”)
devtools::install_github("ekstroem/CoOL")
```

In order to plot dendrograms, install ggtree:
```{r}
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("ggtree")
```