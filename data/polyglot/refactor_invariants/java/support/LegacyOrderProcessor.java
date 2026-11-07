import java.util.ArrayList;
import java.util.List;

/** The original implementation, kept for differential testing. */
public final class LegacyOrderProcessor {
    public static List<ProcessedOrder> processOrderData(List<OrderRecord> records) {
        List<ProcessedOrder> res = new ArrayList<>();
        if (records != null) {
            for (OrderRecord r : records) {
                if (r != null && r.qty() != null && r.price() != null) {
                    if (r.qty() > 0 && r.price() >= 0) {
                        double total = r.qty() * r.price();
                        if (r.discount() != null && r.discount() > 0 && r.discount() <= 1) {
                            total *= (1 - r.discount());
                        }
                        res.add(new ProcessedOrder(r.id() != null ? r.id() : "N/A", total));
                    }
                }
            }
        }
        return res;
    }
}
