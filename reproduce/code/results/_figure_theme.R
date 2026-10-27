## _figure_theme.R ------------------------------------------------------------
## Makes the type in the thesis figures bigger on the page, WITHOUT changing
## what the figures look like.
##
## THE PROBLEM
## -----------
##     printed pt  =  base_size  x  placed width / saved width
##
## theme_bench(base_size = 8) is the right scale for the 6.50 in column, but
## every plot is ggsave()d onto a 130-381 mm canvas and then scaled by hand in
## Google Slides into a panel slot 1.5-3.4 in wide. 8 pt x (2.29/12.01) = 1.5 pt
## on paper. Raising base_size cannot fix it: whatever it is set to, it is still
## divided by ~5 on the way to the page. The saved canvas is the only lever.
##
## THE TRADE, WHICH IS REAL
## ------------------------
## Shrinking the canvas makes the type bigger relative to the panel -- but the
## axis labels, strips, legend and margins grow by the same factor, and they
## grow out of the panel. Shrink far enough and the panel collapses: at the full
## "8 pt on the page" correction, Figure 2a's furniture is TALLER than its own
## 1.50 in slot and the facet strips stack with no plot between them.
##
## So this is a deliberately GENTLE correction. FIG_K = 0.55 was chosen by
## rendering each plot at 0.45 / 0.55 / 0.65 / 0.75 / 0.85 and comparing against
## the pre-change originals in results/_png_backup_pre_typography_2026-08-19/:
## 0.55 is the largest gain at which every plot still reads as the same figure.
## 0.45 is where y tick labels start to touch and panels visibly compress.
##
## Net effect: type prints ~1.8x larger (1.2-2.7 pt -> 2.1-4.3 pt) and the plots
## are otherwise indistinguishable from the originals.
##
## HOW TO USE
## ----------
##   source("../_figure_theme.R")   # beside source("../_cluster_dependent.R")
##   FIG_ARM <- "P4"                # beside PLOT_DIR
##   save_fig("plot1_....png", p1, width = 330, height = 152,
##            units = "mm", dpi = 1200, path = PLOT_DIR)   # was ggsave(...)
##
## save_fig() is a drop-in for ggsave(). A plot NOT in FIG_K_PLOTS gets
## scale = 1 -- byte-identical to today -- so the ~290 exploratory plots and
## everything outside Figures 2-7 are untouched.
## -----------------------------------------------------------------------------

library(ggplot2)

PT_BASE <- 8      # what theme_bench(base_size=) is, for the pt reporting only

## THE KNOB ---------------------------------------------------------------------
## Canvas multiplier. 1.00 = the original figure, exactly. Lower = larger type
## on the page and less room for the plot. Verified range: 0.85 (barely any
## gain) down to 0.55 (still looks like the original). Below ~0.50 the dense
## panels start to compress visibly -- check the render, do not just trust this.
FIG_K <- 0.55

## Extra shrink on POINT sizes only, on top of k. Proportional dots (1.00) are
## the same size on the page as the original, but the panel is no longer the
## same shape, and on the dense jitters they merge into a blob. 0.70 keeps the
## jitter reading as a distribution with the boxplot visible underneath.
## Applies to `size` on non-text layers; linewidths stay at k.
FIG_MARK <- 0.70

## Height. THE PROBLEM: the plot's furniture -- axis labels, strips, legend,
## margins -- is a FIXED number of mm whatever k is (42 mm on plot1, 77 mm on
## plot7), because it is at print size by construction. So shrinking the canvas
## takes the height out of the PANEL, not out of the furniture: plot1's panel
## goes from 27.4 mm to 10.3 mm and the boxplots squash until you cannot read
## them against the y axis.
##
## The fix is to give the figure back the height the bigger furniture needs.
## Keeping the panel the same size on the page as the original requires
##
##     stretch = 1 + (F / h) * (1/k - 1)      F = furniture height, mm
##
## which is measured per plot at save time, so it self-tunes: 1.23 on plot1
## (furniture-heavy), 1.05 on plot31 (furniture-light). Set FIG_STRETCH to FALSE
## to go back to a pure proportional shrink.
##
## NOTE this changes the ASPECT RATIO -- the only thing in this file that does.
## Slides "Replace image" will keep the old frame, so the affected panels need
## their frame height adjusted once. save_fig() reports the new aspect for each.
FIG_STRETCH     <- TRUE
FIG_STRETCH_MAX <- 1.60   # never more than this much taller

## Page geometry ---------------------------------------------------------------
FIG_W_FULL <- 6.50   # in -- full text column (build_manuscript.js CONTENT_PX 624)
FIG_H_MAX  <- 8.44   # in -- above this the builder clamps the whole figure

## Unit conversions ------------------------------------------------------------
## Helpers, for hand-written sizes in the scripts. Nothing here calls them --
## save_fig() rescales existing sizes rather than restating them -- but they are
## the correct conversions if you ever need to set one explicitly.
##   element_text(size=)      is in POINTS
##   geom_text(size=)         is in MILLIMETRES     -> mm(6) is a 6 pt label
##   element_line(linewidth=) is ggplot's own unit  -> lw(0.5) is a 0.5 pt rule
mm <- function(pt) pt / .pt
lw <- function(pt) pt * (96 / 72) / .pt

## -----------------------------------------------------------------------------
## Which plots are corrected: the panels that reach Figures 2-7.
## Keyed "<arm>/<filename>" because filenames collide across arms --
## plot2_tool_heatmap_kappa.png exists in P4, P5, P6a, P6b and P8b.
##
## P9/Plot1_heatmap_kappa.png (Figure 5b) took three attempts. As it stood it
## could not be corrected at any factor -- patchwork's `guides = "collect"` put
## its two legends on the RIGHT, against the legend.position = "bottom" that
## both components ask for, leaving the heatmap ~40 % of the width and the block
## headers colliding by k = 0.90. Moving the collected legends back to the
## bottom (one line in Plots_Analysis.R) lets it take the full 0.55.
## -----------------------------------------------------------------------------
FIG_K_PLOTS <- c(
  ## Figure 2
  "P4/plot1_category_kappa_overview.png",  "P5/plot1_category_kappa_overview.png",
  "P4/plot2_tool_heatmap_kappa.png",       "P5/plot2_tool_heatmap_kappa.png",
  ## Figure 3
  "P4/plot7_runtime_by_category.png",      "P5/plot7_runtime_by_category.png",
  "P4/plot8_memory_by_category.png",       "P5/plot8_memory_by_category.png",
  ## Figure 4 a-c
  "P4/plot24_category_factor_sensitivity.png",
  "P5/plot24_category_factor_sensitivity.png",
  "P4/plot34_runtime_factor_sensitivity_by_category.png",
  "P5/plot34_runtime_factor_sensitivity_by_category.png",
  "P4/plot38_memory_factor_sensitivity_by_category.png",
  "P5/plot38_memory_factor_sensitivity_by_category.png",
  ## Figure 4 d-f
  "P4/plot31_knn_purity_by_category.png",  "P5/plot31_knn_purity_by_category.png",
  "P4/plot37_cellcount_vs_runtime_by_category.png",
  "P5/plot37_cellcount_vs_runtime_by_category.png",
  "P4/plot39_cellcount_vs_memory_by_category.png",
  "P5/plot39_cellcount_vs_memory_by_category.png",
  ## Figure 5b
  "P9/Plot1_heatmap_kappa.png",
  ## Figure 6
  "P7a/plot1_heatmap_scores.png",          "P7b/plot1_heatmap_scores.png",
  "P7a/plot6_dataset_characteristics.png", "P7b/plot6_dataset_characteristics.png",
  ## Figure 7
  "P8b/plot2_tool_heatmap_kappa.png",
  ## Phase 1 vs Phase 2 Wilcoxon (P4_vs_P5_Wilcoxon.R). Not yet placed in any
  ## figure, so FIG_PLACED has no entry and the printed-pt column reads "--".
  ## Everything else applies unchanged: the correction is a canvas multiplier,
  ## it does not depend on where the panel lands.
  "P5/P1vP2_wilcoxon_paired_dotplot.png",
  "P5/P1vP2_wilcoxon_diff_boxplot.png",
  ## Added on request; not currently placed in a numbered figure either.
  "P7b/plot11_chars_vs_runtime.png",
  ## P7a has no "plot11_chars_vs_runtime" -- plot19 is the counterpart of P7b's
  ## plot11 by name and size, plot11_runtime_tradeoff matches by number. Both
  ## registered so either reading is covered.
  "P7a/plot19_chars_vs_runtime.png",
  "P7a/plot11_runtime_tradeoff.png",
  "P8b/plot14_main_effects.png",
  "P8b/plot33_runtime_vs_characteristics.png",
  ## Batch 4. Same treatment, none of them yet placed in a numbered figure.
  "P4/plot3_kappa_runtime_tradeoff.png",   "P5/plot3_kappa_runtime_tradeoff.png",
  "P4/plot4_kappa_memory_tradeoff.png",    "P5/plot4_kappa_memory_tradeoff.png",
  "P8b/plot1_tool_kappa_overview.png",
  "P6a/plot2_tool_heatmap_kappa.png",      "P6b/plot2_tool_heatmap_kappa.png",
  ## Plots_Analysis_OrthogonalFactors.R -- a separate script in the P4 folder
  ## writing to ./Plots_OrthogonalFactors. It sets FIG_ARM <- "P4" as well; the
  ## three filenames are unique repo-wide, so the shared prefix cannot collide.
  "P4/eta2_continuous_vs_categorical.png",
  "P4/mediation_purity_by_setfactors.png",
  "P4/proxy_orthogonality_correlation.png",
  ## Batch 5. Asked for as "plot7_runtime_by_tool (p9)", but P9 has no file of
  ## that name -- every P9 plot is capitalised Plot<N>_<something> and none of
  ## them is runtime-by-tool. The exact filename lives in P6a, P6b, P8a and P8b.
  ## Both readings covered: the exact name in the three arms already wired, and
  ## P9's two candidates (Plot7 by number, Plot11 as the arm's runtime plot).
  ## P8a also has plot7_runtime_by_tool but has never been wired or in scope.
  "P6a/plot7_runtime_by_tool.png", "P6b/plot7_runtime_by_tool.png",
  "P8b/plot7_runtime_by_tool.png",
  "P9/Plot7_batch_robustness_slope.png",
  "P9/Plot11_runtime_scatter.png"
)

## Per-plot deviation from FIG_K, where a plot needs to stay closer to the
## original than the global value allows.
##
##   P8b/plot2 is 254 x 127 mm -- unusually SHORT for its width, so its fixed
##   furniture (rotated dataset labels ~28 mm at print size, plus the colourbar)
##   is already a large share of the height. At k = 0.55 the five model rows
##   compress into a thin band and the row labels collide. Rendered at 0.90 /
##   0.85 / 0.80 / 0.75 / 0.65: clean down to 0.75, compressing by 0.65.
FIG_K_OVERRIDE <- c(
  "P8b/plot2_tool_heatmap_kappa.png" = 0.75
)

## -----------------------------------------------------------------------------
## The one unavoidable layout change, on two plots.
##
## plot7 / plot8 are 130 mm wide and their five-key legend fills that width on
## one row. At ANY shrink it is cut off at both edges -- verified at 0.85, 0.75,
## 0.65, 0.55 and 0.45, and with 2- and 3-row wraps. One key per row, with the
## title above rather than inline, is the only arrangement that fits.
## -----------------------------------------------------------------------------
FIG_LEGEND_ROWS <- c(
  "P4/plot7_runtime_by_category.png" = 5, "P5/plot7_runtime_by_category.png" = 5,
  "P4/plot8_memory_by_category.png"  = 5, "P5/plot8_memory_by_category.png"  = 5,
  ## plot3 carries a size legend AND a 5-key Tool Category legend on one row;
  ## together they run past the 153 mm canvas and "Semi-Supervised" is cut off.
  ## Two rows plus the stacked guide boxes .stack_legend() now applies.
  ## plot4 has only the colour legend and measured a 1.8 % right margin, so it
  ## is deliberately NOT listed.
  "P4/plot3_kappa_runtime_tradeoff.png" = 2, "P5/plot3_kappa_runtime_tradeoff.png" = 2,
  ## P9/Plot7 draws FOUR guide boxes on one row -- Tool Category plus a
  ## linewidth, an alpha and a size legend, all three keyed to the same
  ## `highlight` variable and all three left on their default guide. Measured
  ## 198 mm of legend against a 140 mm canvas. Stacking the boxes (which
  ## .stack_legend now does) plus two rows on the colour keys brings it inside.
  "P9/Plot7_batch_robustness_slope.png" = 2
)

## Opt-in title/subtitle/caption wrapping -- see .wrap_titles() below for why it
## is opt-in rather than global. Add a key only when the render shows ink at the
## canvas edge; the measurement is
##   python3 -c "...": ink in the last 3 px columns of the PNG.
##
##   proxy_orthogonality_correlation is the narrowest canvas in the project
##   (160 -> 88 mm) and its 116-character subtitle needs ~143 mm at 7 pt.
##   mediation_purity_by_setfactors (200 -> 110 mm) clips its title band too,
##   and so does eta2 (254 -> 140 mm), whose subtitle stops 4 px short of the
##   edge mid-word. Test for a real MARGIN (>= 1 % of width), not for ink in the
##   last few pixel columns -- eta2 passed an exact-edge check while visibly
##   truncating "Concordance" to "Concordan".
FIG_WRAP <- c(
  "P4/proxy_orthogonality_correlation.png",
  "P4/mediation_purity_by_setfactors.png",
  "P4/eta2_continuous_vs_categorical.png"
)

## Printed width on the page, in mm (figure_panel_map.md). Reporting only.
FIG_PLACED <- c(
  "P4/plot1_category_kappa_overview.png" = 83, "P5/plot1_category_kappa_overview.png" = 83,
  "P4/plot2_tool_heatmap_kappa.png" = 66,      "P5/plot2_tool_heatmap_kappa.png" = 66,
  "P4/plot7_runtime_by_category.png" = 38,     "P5/plot7_runtime_by_category.png" = 38,
  "P4/plot8_memory_by_category.png" = 38,      "P5/plot8_memory_by_category.png" = 38,
  "P4/plot24_category_factor_sensitivity.png" = 58,
  "P5/plot24_category_factor_sensitivity.png" = 58,
  "P4/plot34_runtime_factor_sensitivity_by_category.png" = 58,
  "P5/plot34_runtime_factor_sensitivity_by_category.png" = 58,
  "P4/plot38_memory_factor_sensitivity_by_category.png" = 58,
  "P5/plot38_memory_factor_sensitivity_by_category.png" = 58,
  "P4/plot31_knn_purity_by_category.png" = 56, "P5/plot31_knn_purity_by_category.png" = 56,
  "P4/plot37_cellcount_vs_runtime_by_category.png" = 56,
  "P5/plot37_cellcount_vs_runtime_by_category.png" = 56,
  "P4/plot39_cellcount_vs_memory_by_category.png" = 56,
  "P5/plot39_cellcount_vs_memory_by_category.png" = 56,
  "P7a/plot1_heatmap_scores.png" = 78,         "P7b/plot1_heatmap_scores.png" = 78,
  "P7a/plot6_dataset_characteristics.png" = 76,"P7b/plot6_dataset_characteristics.png" = 76,
  "P8b/plot2_tool_heatmap_kappa.png" = 86,
  "P9/Plot1_heatmap_kappa.png" = 56
)

## -----------------------------------------------------------------------------
## Marks must shrink with the canvas; TEXT must not.
##
## geom_point(size=), geom_jitter(size=) and linewidth= are millimetres on the
## device, exactly like geom_text(size=). Text is the thing we are deliberately
## holding at print size. A data mark is not: it is sized relative to its panel,
## and the panel just shrank. Left alone, a 4 mm dot drawn for a 381 mm canvas
## keeps its 4 mm on a 210 mm one and swallows the boxplot underneath.
##
## Multiplying every non-text layer's explicit size/linewidth/stroke by the same
## k restores each mark to exactly the printed size it had in the original.
## Geom DEFAULTS are left alone: under ggplot2 4.x they resolve from_theme(), so
## a boxplot outline is already a correct print-weight line.
## -----------------------------------------------------------------------------
## In-plot TEXT is a third case, between the two.
##
## Axis / strip / legend text is theme text: it has the margin to itself and is
## held at print size -- that is the whole point of this file. A geom_text is
## not: it sits INSIDE the data geometry (a heatmap cell, the gap between five
## dodged bars) and that geometry shrinks with the canvas. Held at full size it
## collides -- at k = 0.55 the five "ns" markers on Figure 4a overlap and the
## in-cell kappa values on Figure 2b crowd into each other.
##
## sqrt(k) splits the difference: the annotation grows ~1.35x on the page while
## its share of the cell goes from 17 % (original) to 24 % rather than 32 %.
## Verified by rendering plot2 and plot24 at both sqrt(k) and k.
.TEXT_GEOMS <- c("GeomText", "GeomLabel", "GeomTextRepel", "GeomLabelRepel")

.scale_layer <- function(l, k) {
  isText <- any(class(l$geom) %in% .TEXT_GEOMS)
  m <- if (isText) sqrt(k) else k
  ch <- list()
  for (par in c("size", "linewidth", "stroke")) {
    v <- l$aes_params[[par]]
    if (is.null(v) || !is.numeric(v)) next
    ## `size` on a non-text layer is a point size -> also apply FIG_MARK
    ch[[par]] <- v * m * (if (!isText && par == "size") FIG_MARK else 1)
  }
  if (!length(ch)) return(l)
  out <- ggproto(NULL, l)                          # child shadows the parent;
  out$aes_params <- modifyList(l$aes_params, ch)   # the original is untouched
  out
}

## A colourbar's barwidth is absolute too: plot2 asks for unit(10, "cm"), which
## was 39 % of a 254 mm canvas. ggplot2 4.x keeps barwidth/barheight as
## legend.key.width/height inside the guide's own theme.
## Applied by ADDING a replacement guide -- never by cloning the Scale, because
## ggproto(NULL, <Scale>) makes a child whose super chain recurses and blows the
## node stack at render time.
.scale_guides <- function(p, k) {
  if (is.null(p$scales)) return(p)
  add <- list()
  for (sc in p$scales$scales) {
    g <- sc$guide
    if (!inherits(g, "GuideColourbar") || is.null(g$params$theme)) next
    th <- g$params$theme
    kw <- th$legend.key.width; kh <- th$legend.key.height
    if (!(grid::is.unit(kw) || grid::is.unit(kh))) next
    add[[sc$aesthetics[[1]]]] <- guide_colourbar(theme = theme(
      legend.key.width      = if (grid::is.unit(kw)) kw * k else NULL,
      legend.key.height     = if (grid::is.unit(kh)) kh * k else NULL,
      legend.title.position = th$legend.title.position %||% "top"))
  }
  if (!length(add)) return(p)
  p + do.call(guides, add)
}

## A `size` or `linewidth` SCALE carries millimetres too, in a place aes_params
## never looks.
## scale_size_continuous(range = c(2, 8)) is 2-8 mm on the device, exactly like
## geom_point(size = 3.5) -- but the range is captured inside pal_area() at
## construction, so there is no field to read and nothing for .scale_layer() to
## find. Left alone, P4/P5 plot3's bubbles keep their 8 mm on a canvas that just
## went from 279 to 153 mm, which is the "massive data points" failure again.
##
## The palette is a plain function returning mm (verified: for both
## scale_size_continuous and scale_size_manual, palette(c(0, 1)) hands back the
## range endpoints unchanged), so wrapping it is the whole fix -- no
## ggproto(NULL, <Scale>), which recurses through `super` and blows the node
## stack. Wrapping mutates the Scale in place, so a flag makes it idempotent:
## saving the same plot object twice must not square the multiplier.
##
## Note this BREAKS the non-mutation invariant .scale_layer() keeps with
## ggproto(NULL, l): the caller's plot object is modified. Safe today because
## every scale_size_*() call in this repo is constructed inline in its own
## `pN <- ... + scale_size_*(...)` chain, so no two plots share a Scale. If a
## shared scale object is ever hoisted into a variable, registered and
## unregistered plots would both shrink -- clone at that point.
## .shrunk_palette() exists so each wrapper gets its OWN frame holding its own
## `p0`. Built inline in the loop instead, every closure would close over the
## single loop frame, so a plot with two size scales would leave both pointing
## at the LAST p0 -- and the second wrap would find p0 already wrapped and
## recurse forever.
## force(p0) is load-bearing: unforced it is a promise for `sc$palette`, which
## by the time the wrapper runs IS the wrapper -- infinite recursion at render.
.shrunk_palette <- function(p0, m) { force(p0); force(m); function(x) p0(x) * m }

## `linewidth` scales are the same story as `size` scales -- P9's p7 sets its
## slope weights with scale_linewidth_manual(values = c(0.5, 1.5, 1.5)), which is
## millimetres exactly like an aes_params linewidth. Same multiplier split as
## .scale_layer() uses: point sizes take FIG_MARK on top of k, line weights do
## not. `alpha` scales are unitless and must be left alone.
.MARK_SCALE_MULT <- list(size = function(k) k * FIG_MARK,
                         linewidth = function(k) k)

.scale_mark_scales <- function(p, k) {
  if (is.null(p$scales)) return(p)
  for (sc in p$scales$scales) {
    aes <- intersect(sc$aesthetics, names(.MARK_SCALE_MULT))
    if (!length(aes)) next
    if (!is.function(sc$palette)) next
    if (isTRUE(sc$.fig_scaled)) next
    sc$palette     <- .shrunk_palette(sc$palette, .MARK_SCALE_MULT[[aes[1]]](k))
    sc$.fig_scaled <- TRUE
  }
  p
}

## -----------------------------------------------------------------------------
## Title / subtitle wrapping (opt-in, FIG_WRAP).
##
## A title is theme text, so it is deliberately held at print size -- but unlike
## axis text it does not shrink to fit and ggplot does not wrap it. On a wide
## canvas that is invisible; at 88 mm the 116-character subtitle on
## proxy_orthogonality_correlation runs 55 mm past the right edge and is cut.
##
## Wrapping is typesetting, not a content edit: identical words, more lines.
## It is opt-in because it changes the furniture height, hence the stretch,
## hence the aspect ratio -- and the panels already delivered have their Slides
## frame heights set from the aspects they have now. Only add a key here when
## the render actually shows ink at the canvas edge.
##
## Widths are MEASURED with grobWidth on a device of the true saved size, not
## estimated from a characters-per-line rule: the strings mix bold, regular and
## Unicode (kappa, eta-squared, arrows) and a character estimate is wrong by
## enough to either clip anyway or wrap a line that would have fit.
.wrap_text <- function(txt, el, avail_mm) {
  if (is.null(txt) || !is.character(txt) || !length(txt) || !nzchar(txt[1])) return(txt)
  fs  <- el$size %||% 8
  ff  <- el$face %||% "plain"
  fam <- el$family %||% ""
  wmm <- function(st) grid::convertWidth(grid::grobWidth(grid::textGrob(
           st, gp = grid::gpar(fontsize = fs, fontface = ff, fontfamily = fam))),
           "mm", valueOnly = TRUE)
  out <- vapply(strsplit(txt[1], "\n", fixed = TRUE)[[1]], function(para) {
    if (!nzchar(para) || wmm(para) <= avail_mm) return(para)
    cur <- ""; lines <- character(0)
    for (w in strsplit(para, " ", fixed = TRUE)[[1]]) {
      cand <- if (nzchar(cur)) paste(cur, w) else w
      if (nzchar(cur) && wmm(cand) > avail_mm) { lines <- c(lines, cur); cur <- w }
      else cur <- cand
    }
    paste(c(lines, cur), collapse = "\n")
  }, character(1), USE.NAMES = FALSE)
  paste(out, collapse = "\n")
}

.wrap_titles <- function(p, w_mm, h_mm) {
  if (!inherits(p, "ggplot")) return(p)
  th <- tryCatch(ggplot2:::plot_theme(p), error = function(e) NULL)
  if (is.null(th)) return(p)

  m <- tryCatch(ggplot2::calc_element("plot.margin", th), error = function(e) NULL)
  side_mm <- if (grid::is.unit(m))
    sum(grid::convertWidth(m[c(2, 4)], "mm", valueOnly = TRUE)) else 0
  avail <- w_mm - side_mm - 1                      # 1 mm of slack for rounding

  f <- tempfile(fileext = ".png")
  grDevices::png(f, width = w_mm / 25.4, height = h_mm / 25.4,
                 units = "in", res = 100, type = "cairo")
  on.exit({grDevices::dev.off(); unlink(f)}, add = TRUE)

  new <- list()
  for (slot in c("title", "subtitle", "caption")) {
    txt <- p$labels[[slot]]
    if (is.null(txt)) next
    el <- tryCatch(ggplot2::calc_element(paste0("plot.", slot), th),
                   error = function(e) NULL)
    if (is.null(el)) next
    wrapped <- .wrap_text(txt, el, avail)
    if (!identical(wrapped, txt)) new[[slot]] <- wrapped
  }
  if (!length(new)) return(p)
  ## Both positions must be "plot". By default ggplot aligns title/subtitle to
  ## the PANEL, so they start inset by the width of the y-axis labels -- on this
  ## heatmap "No. Cell Types" alone is 20 mm of an 88 mm canvas. Wrapping to the
  ## canvas width while the text is drawn from the panel's left edge still
  ## clips; anchoring to the canvas is what makes the measurement true.
  p + do.call(labs, new) +
    theme(plot.title.position = "plot", plot.caption.position = "plot")
}

.scale_marks <- function(p, k) {
  if (!inherits(p, "ggplot")) return(p)
  if (!is.null(p$patches$plots))              # patchwork: recurse into children
    p$patches$plots <- lapply(p$patches$plots, .scale_marks, k = k)
  if (!is.null(p$layers))
    p$layers <- lapply(p$layers, .scale_layer, k = k)
  p <- .scale_mark_scales(p, k)
  .scale_guides(p, k)
}

## Fixed (non-panel) height of a plot, in mm, at a given canvas size. Every row
## of a ggplot gtable is a fixed unit except the panels, which are "null".
.furniture_mm <- function(p, w_mm, h_mm) {
  f <- tempfile(fileext = ".png")
  grDevices::png(f, width = w_mm / 25.4, height = h_mm / 25.4,
                 units = "in", res = 100, type = "cairo")
  on.exit({grDevices::dev.off(); unlink(f)}, add = TRUE)
  g  <- tryCatch(ggplotGrob(p), error = function(e) NULL)
  if (is.null(g)) return(NA_real_)
  hs <- g$heights
  sum(grid::convertHeight(hs[!grepl("null", as.character(hs))],
                          "in", valueOnly = TRUE)) * 25.4
}

## Width the legends need, in mm, at a given canvas size. The companion to
## .furniture_mm(): that one asks "how much HEIGHT is not panel", this one asks
## "is the guide box wider than the canvas it has to fit in".
##
## Worth having as a permanent column on the [fig] line. The pixel test on the
## saved PNG only sees the CONSEQUENCE of an overflowing legend, and only when
## the overflow happens to put ink in the last column -- plot3's clipped
## "Semi-Supervised" key registered, but a legend that overflows into blank
## space between keys would not. This is the number that actually discriminates.
.guide_mm <- function(p, w_mm, h_mm) {
  f <- tempfile(fileext = ".png")
  grDevices::png(f, width = w_mm / 25.4, height = h_mm / 25.4,
                 units = "in", res = 100, type = "cairo")
  on.exit({grDevices::dev.off(); unlink(f)}, add = TRUE)
  g <- tryCatch(ggplotGrob(p), error = function(e) NULL)
  if (is.null(g)) return(NA_real_)
  i <- grep("guide-box", g$layout$name)
  if (!length(i)) return(0)
  max(vapply(i, function(j) {
    gr <- g$grobs[[j]]
    w  <- tryCatch(grid::convertWidth(grid::grobWidth(gr), "mm", valueOnly = TRUE),
                   error = function(e) NA_real_)
    if (is.na(w) || w <= 0)
      w <- tryCatch(sum(grid::convertWidth(gr$widths, "mm", valueOnly = TRUE)),
                    error = function(e) NA_real_)
    if (is.na(w)) 0 else w
  }, numeric(1)))
}

.stretch_for <- function(p, w_mm, h_mm, k) {
  if (!isTRUE(FIG_STRETCH) || k >= 1) return(1)
  F <- .furniture_mm(p, w_mm * k, h_mm * k)
  if (is.na(F) || F <= 0 || F >= h_mm) return(1)
  min(FIG_STRETCH_MAX, max(1, 1 + (F / h_mm) * (1 / k - 1)))
}

.stack_legend <- function(p, nrow) {
  if (!inherits(p, "ggplot") || is.null(p$scales)) return(p)
  disc <- unlist(lapply(p$scales$scales, function(sc)
    if (inherits(sc, "ScaleDiscrete")) sc$aesthetics else NULL))
  disc <- intersect(unique(disc), c("fill", "colour", "color", "shape"))
  if (!length(disc)) return(p)
  g <- guide_legend(nrow = nrow, theme = theme(legend.title.position = "top"))
  ## legend.box = "vertical" as well: nrow only reshapes the discrete guide, and
  ## a plot with TWO guide boxes (plot3 has a size legend beside the colour one)
  ## still lays them out side by side and still overflows. No-op where there is
  ## only one box, which is every other plot in FIG_LEGEND_ROWS.
  p + do.call(guides, stats::setNames(rep(list(g), length(disc)), disc)) +
    theme(legend.box = "vertical")
}

## -----------------------------------------------------------------------------
## Internals
## -----------------------------------------------------------------------------
.fig_arm <- function()
  if (exists("FIG_ARM", envir = globalenv())) get("FIG_ARM", envir = globalenv()) else NA_character_

.to_mm <- function(x, units)
  switch(units, "mm" = x, "cm" = x * 10, "in" = x * 25.4,
         stop("save_fig(): unsupported units '", units, "'"))

.fig_plan <- function(filename, width, units = "mm", arm = .fig_arm()) {
  if (is.na(arm)) return(NULL)
  key <- paste0(arm, "/", filename)
  if (!key %in% FIG_K_PLOTS) return(NULL)
  k <- if (key %in% names(FIG_K_OVERRIDE)) unname(FIG_K_OVERRIDE[[key]]) else FIG_K
  w_mm <- .to_mm(width, units)
  placed <- if (key %in% names(FIG_PLACED)) unname(FIG_PLACED[[key]]) else NA_real_
  list(key = key, k = k, w_mm = w_mm, saved_mm = w_mm * k,
       was_pt = PT_BASE * placed / w_mm, now_pt = PT_BASE * placed / (w_mm * k))
}

## -----------------------------------------------------------------------------
## save_fig() -- drop-in replacement for ggsave()
## -----------------------------------------------------------------------------
save_fig <- function(filename, plot = last_plot(), width, height,
                     units = "mm", dpi = 600, path = ".", ...) {

  p <- .fig_plan(filename, width, units)

  if (is.null(p)) {                       # not a figure panel: unchanged
    ggsave(filename, plot, width = width, height = height,
           units = units, dpi = dpi, path = path, ...)
    return(invisible(filename))
  }

  out <- .scale_marks(plot, p$k)
  if (p$key %in% names(FIG_LEGEND_ROWS))
    out <- .stack_legend(out, unname(FIG_LEGEND_ROWS[[p$key]]))

  h_mm <- .to_mm(height, units)
  ## Wrap BEFORE the stretch is measured -- wrapping adds title lines, which are
  ## furniture, and .stretch_for() has to see the height they actually take.
  if (p$key %in% FIG_WRAP)
    out <- .wrap_titles(out, p$w_mm * p$k, h_mm * p$k)
  st   <- .stretch_for(out, p$w_mm, h_mm, p$k)

  saved_w_mm <- p$w_mm * p$k
  saved_h_mm <- h_mm  * p$k * st
  if (saved_h_mm / 25.4 > FIG_H_MAX)
    warning(sprintf("%s: saved height %.2f in exceeds %.2f in -- the builder will clamp it.",
                    filename, saved_h_mm / 25.4, FIG_H_MAX))

  ggsave(filename, out, width = saved_w_mm, height = saved_h_mm,
         units = "mm", dpi = dpi, path = path, ...)

  pt_txt <- if (is.na(p$was_pt)) "  -- not yet placed --"
            else sprintf("%.1f -> %.1f pt", p$was_pt, p$now_pt)
  gmm <- .guide_mm(out, saved_w_mm, saved_h_mm)
  gtx <- if (is.na(gmm)) "legend --"
         else sprintf("legend %3.0f/%3.0fmm%s", gmm, saved_w_mm,
                      if (gmm > saved_w_mm) " OVERFLOW" else "")
  message(sprintf("  [fig] %-44s %5.0fx%-5.0f -> %4.0fx%-4.0f mm (k %.2f, h x%.2f)  %-20s aspect %.3f -> %.3f  %s",
                  filename, p$w_mm, h_mm, saved_w_mm, saved_h_mm, p$k, st,
                  pt_txt, h_mm / p$w_mm, saved_h_mm / saved_w_mm, gtx))
  invisible(filename)
}

## Reporting only -- save_fig() uses the real `width` argument it is passed, so
## nothing below can affect a render. Do not wire it into the render path.
## Dry run: the whole table without rendering anything.
##   Rscript -e 'source("_figure_theme.R"); fig_report()'
FIG_NOMINAL <- c(
  "plot1_category_kappa_overview.png" = 330, "plot2_tool_heatmap_kappa.png" = 254,
  "plot7_runtime_by_category.png" = 130,     "plot8_memory_by_category.png" = 130,
  "plot24_category_factor_sensitivity.png" = 305,
  "plot34_runtime_factor_sensitivity_by_category.png" = 305,
  "plot38_memory_factor_sensitivity_by_category.png" = 305,
  "plot31_knn_purity_by_category.png" = 381,
  "plot37_cellcount_vs_runtime_by_category.png" = 381,
  "plot39_cellcount_vs_memory_by_category.png" = 381,
  "plot1_heatmap_scores.png" = 254, "plot6_dataset_characteristics.png" = 305,
  "P1vP2_wilcoxon_paired_dotplot.png" = 356, "P1vP2_wilcoxon_diff_boxplot.png" = 305,
  "plot11_chars_vs_runtime.png" = 254, "plot14_main_effects.png" = 305,
  "plot33_runtime_vs_characteristics.png" = 305,
  "plot19_chars_vs_runtime.png" = 254, "plot11_runtime_tradeoff.png" = 203,
  "Plot1_heatmap_kappa.png" = 279,
  ## Batch 4. plot2_tool_heatmap_kappa.png = 254 above already covers P6a/P6b:
  ## all four arms that draw it use the same nominal width.
  "plot3_kappa_runtime_tradeoff.png" = 279, "plot4_kappa_memory_tradeoff.png" = 279,
  "plot1_tool_kappa_overview.png" = 330,
  "eta2_continuous_vs_categorical.png" = 254,
  "mediation_purity_by_setfactors.png" = 200,
  "proxy_orthogonality_correlation.png" = 160,
  ## Batch 5
  "plot7_runtime_by_tool.png" = 330,
  "Plot7_batch_robustness_slope.png" = 254, "Plot11_runtime_scatter.png" = 356
)
FIG_NOMINAL["P8b/plot2_tool_heatmap_kappa.png"] <- 254

fig_report <- function() {
  cat(sprintf("\nFIG_K = %.2f   (1.00 = the original figure, exactly)\n\n", FIG_K))
  cat(sprintf("%-54s %8s %8s %6s %8s %8s\n",
              "key", "nominal", "saved", "k", "was pt", "now pt"))
  cat(strrep("-", 98), "\n", sep = "")
  for (key in FIG_K_PLOTS) {
    f   <- sub("^[^/]+/", "", key)
    nom <- if (key %in% names(FIG_NOMINAL)) FIG_NOMINAL[[key]] else FIG_NOMINAL[[f]]
    k   <- if (key %in% names(FIG_K_OVERRIDE)) unname(FIG_K_OVERRIDE[[key]]) else FIG_K
    pl  <- if (key %in% names(FIG_PLACED)) unname(FIG_PLACED[[key]]) else NA_real_
    cat(sprintf("%-54s %6.0fmm %6.0fmm %6.2f %8s %8s\n", key, nom, nom * k, k,
                if (is.na(pl)) "--" else sprintf("%.1f", PT_BASE * pl / nom),
                if (is.na(pl)) "--" else sprintf("%.1f", PT_BASE * pl / (nom * k))))
  }
  cat("\n'was pt' = what base_size printed at before; 'now pt' = after.\n\n")
  invisible(NULL)
}
