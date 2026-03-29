for node in $(sinfo -N -h -o "%N" | sort -u); do
    result=$(srun --nodelist=$node -N1 -n1 --time=00:01:00 which fakeroot 2>/dev/null)
    if [ -n "$result" ]; then
        echo "Node $node has fakeroot: $result"
    fi
done