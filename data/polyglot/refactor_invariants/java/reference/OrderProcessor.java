import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class OrderProcessor {
    private OrderProcessor() {}

    public static List<ProcessedOrder> processOrderData(List<OrderRecord> records) {
        if (records == null) {
            return new ArrayList<>();
        }
        List<ProcessedOrder> res = new ArrayList<>();
        for (OrderRecord r : records) {
            if (r == null || r.qty() == null || r.price() == null) continue;
            if (r.qty() <= 0 || r.price() < 0) continue;
            double total = r.qty() * r.price();
            Double d = r.discount();
            if (d != null && d > 0 && d <= 1) {
                total *= (1 - d);
            }
            res.add(new ProcessedOrder(Objects.requireNonNullElse(r.id(), "N/A"), total));
        }
        return res;
    }
}
