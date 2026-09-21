"""Brand palette — black and gold, sampled directly from the My Booth Agent
logo (trident + fish emblem on black). No dependencies."""

BLACK = "#000000"          # page background (matches the logo background exactly)
CHARCOAL = "#161616"       # panels / cards — lifted slightly off black for depth
CHARCOAL_DEEP = "#0D0D0D"  # inputs, deeper panels
GOLD = "#F3BD3B"           # accent — median color sampled from the logo's gold foil
GOLD_DEEP = "#C99A2E"      # darker gold for gradients / hover / pressed states
WHITE = "#FFFFFF"
MUTED = "#B8B8B8"          # secondary text on black

# Kept varied enough that grouped/pie charts stay readable on black —
# an all-gold-and-white chart palette collapses multi-series distinctions.
CHART_COLORWAY = [GOLD, WHITE, "#4FD1C5", "#8E8A92", "#E8734A", GOLD_DEEP]
FONT_STACK = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
