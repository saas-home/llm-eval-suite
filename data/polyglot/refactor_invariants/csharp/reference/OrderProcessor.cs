namespace Orders;

public record Record(string? Id, int Qty, decimal Price, decimal? Discount);
public record ProcessedOrder(string Id, decimal Total);

public static class OrderProcessor {
    public static List<ProcessedOrder> ProcessOrderData(IEnumerable<Record?>? records) {
        if (records is null) return new List<ProcessedOrder>();
        var res = new List<ProcessedOrder>();
        foreach (var r in records) {
            if (r is null) continue;
            if (r.Qty <= 0 || r.Price < 0) continue;
            decimal total = r.Qty * r.Price;
            if (r.Discount is decimal d && d > 0 && d <= 1) {
                total *= 1 - d;
            }
            res.Add(new ProcessedOrder(string.IsNullOrEmpty(r.Id) ? "N/A" : r.Id, Math.Round(total, 2)));
        }
        return res;
    }
}
