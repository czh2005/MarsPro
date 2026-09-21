# Data and claim boundaries

| Layer | What is measured | Supported claim | Unsupported shortcut |
|---|---|---|---|
| Stage 1 labels | broad source/proteome/family association | predicts six environment-related candidate memberships | experimentally proven tolerance for every sequence |
| Stage 1 validation/test | fixed known label cells; unknown cells excluded | catalog-label generalization after validation-only selection | independent biological-function accuracy |
| Stage 2 surface metrics | predicted structure-derived composition | redesign shifts predefined surface features toward reference values | measured stress resistance |
| Stage 2 refolding | ESMFold2 or ESMFold predictions under matched settings | predicted fold confidence and structural similarity | experimental structural validation |
| Stage 2 rescoring | frozen Stage 1 outputs | designs move toward learned six-label patterns | true function, hit rate, or calibrated Mars fitness |

The six outputs are independent. Their equal-weight mean is a post-hoc internal summary used only for paired redesign comparisons, not a separately trained biological endpoint.
