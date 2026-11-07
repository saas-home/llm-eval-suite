import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    static String render(List<ProcessedOrder> out) {
        return out.stream().map(o -> o.id() + ":" + String.format("%.9f", o.total())).collect(Collectors.joining(",", "[", "]"));
    }

    static OrderRecord r(String id, Integer qty, Double price, Double discount) { return new OrderRecord(id, qty, price, discount); }

    public static void main(String[] args) {
        List<List<OrderRecord>> cases = new ArrayList<>();
        cases.add(null);
        cases.add(new ArrayList<>());
        cases.add(Arrays.asList((OrderRecord) null));
        cases.add(List.of(r("a", 2, 3.0, null)));
        cases.add(List.of(r(null, 2, 3.0, null)));
        cases.add(List.of(r("q0", 0, 5.0, null), r("qn", -1, 5.0, null)));
        cases.add(List.of(r("pn", 2, -1.0, null), r("p0", 2, 0.0, null)));
        cases.add(List.of(r("mq", null, 4.0, null), r("mp", 4, null, null)));
        cases.add(List.of(r("d0", 4, 2.5, 0.0), r("d25", 4, 2.5, 0.25)));
        cases.add(List.of(r("d1", 3, 7.0, 1.0), r("d15", 3, 7.0, 1.5), r("dn", 3, 7.0, -0.2)));
        cases.add(List.of(r("r", 7, 1.115, 0.1)));
        for (List<OrderRecord> c : cases) {
            String before = render(LegacyOrderProcessor.processOrderData(c));
            String after = render(OrderProcessor.processOrderData(c));
            if (!before.equals(after)) {
                throw new AssertionError("behavior changed for " + c + ": legacy " + before + " vs refactored " + after);
            }
        }
        System.out.println("POLYGLOT_OK " + System.getenv("EVAL_RUN_ID"));
    }
}
