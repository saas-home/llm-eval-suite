package orders

// legacyProcessOrderData is the original implementation, kept for differential testing.
func legacyProcessOrderData(records []Record) []ProcessedOrder {
	var res []ProcessedOrder
	for _, r := range records {
		if r.Qty > 0 && r.Price >= 0 {
			total := float64(r.Qty) * r.Price
			if r.Discount > 0 && r.Discount <= 1 {
				total *= (1 - r.Discount)
			}
			res = append(res, ProcessedOrder{ID: r.ID, Total: total})
		}
	}
	return res
}
