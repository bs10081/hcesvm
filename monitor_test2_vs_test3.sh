#!/bin/bash
# Monitor Test2 vs Test3 comparison progress

echo "=========================================="
echo "Test2 vs Test3 Comparison Progress Monitor"
echo "=========================================="
echo ""

OUTPUT_FILE="/tmp/claude-1000/-home-bs10081-Developer-hcesvm/tasks/b5674f3.output"

if [ -f "$OUTPUT_FILE" ]; then
    echo "Current Progress:"
    echo "------------------"

    # Count completed datasets
    COMPLETED=$(grep -c "✓.*completed successfully" "$OUTPUT_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "✗.*failed" "$OUTPUT_FILE" 2>/dev/null || echo "0")

    echo "Datasets completed: $COMPLETED"
    echo "Datasets failed: $FAILED"
    echo ""

    # Show current dataset being tested
    CURRENT=$(tail -20 "$OUTPUT_FILE" | grep -E "\[.*\] Testing:" | tail -1)
    if [ ! -z "$CURRENT" ]; then
        echo "Currently testing:"
        echo "$CURRENT"
        echo ""
    fi

    # Show recent progress
    echo "Recent output (last 30 lines):"
    echo "================================"
    tail -30 "$OUTPUT_FILE"
    echo "================================"
    echo ""

    # Show timing info
    STARTED=$(head -20 "$OUTPUT_FILE" | grep "Start time:" | head -1)
    if [ ! -z "$STARTED" ]; then
        echo "$STARTED"
    fi
    echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "Output file not found: $OUTPUT_FILE"
fi

echo ""
echo "To monitor in real-time, run:"
echo "  tail -f $OUTPUT_FILE"
