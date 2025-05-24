import pstats
from pstats import SortKey
p = pstats.Stats("h2h_profile.txt")
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
# We see that get_merge_dst is a very time consuming function, taking 1.6s per call! (unsuprising as it is a janky python for-loop)