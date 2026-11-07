package orders

type Record struct {
	ID       string
	Qty      int
	Price    float64
	Discount float64
}

type ProcessedOrder struct {
	ID    string
	Total float64
}

func ProcessOrderData(records []Record) []ProcessedOrder {
	var res []ProcessedOrder
	for _, r := range records {
		if r.Qty <= 0 || r.Price < 0 {
			continue
		}
		total := float64(r.Qty) * r.Price
		if r.Discount > 0 && r.Discount <= 1 {
			total *= (1 - r.Discount)
		}
		res = append(res, ProcessedOrder{ID: r.ID, Total: total})
	}
	return res
}
