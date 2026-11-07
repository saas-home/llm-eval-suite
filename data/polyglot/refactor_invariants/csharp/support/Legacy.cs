namespace Legacy;

public record Record(string? Id, int Qty, decimal Price, decimal? Discount);
public record ProcessedOrder(string Id, decimal Total);

public static class OrderProcessor {
    public static List<ProcessedOrder> ProcessOrderData(IEnumerable<Record?>? records) {
        var res = new List<ProcessedOrder>();
        if (records != null) {
            foreach (var r in records) {
                if (r != null) {
                    if (r.Qty > 0 && r.Price >= 0) {
                        decimal total = r.Qty * r.Price;
                        if (r.Discount.HasValue) {
                            if (r.Discount.Value > 0 && r.Discount.Value <= 1) {
                                total = total * (1 - r.Discount.Value);
                            }
                        }
                        res.Add(new ProcessedOrder(string.IsNullOrEmpty(r.Id) ? "N/A" : r.Id, Math.Round(total, 2)));
                    }
                }
            }
        }
        return res;
    }
}
