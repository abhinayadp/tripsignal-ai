# Flight Deals and Fare Strategy Guide

A good fare is not only the lowest number. TripSignal should compare total trip value: departure time, arrival time, layover length, baggage, seat selection, cancellation rules, loyalty earning, and risk of a tight connection.

For international trips, TripSignal should check whether a slightly longer layover lowers the fare enough to justify the time. It should warn when a cheap itinerary has an overnight airport stay, a self-transfer, separate tickets, or a terminal change that creates hidden risk.

TripSignal should separate three kinds of savings: cash fare savings, points or miles value, and card benefits. A fare that looks more expensive may be better if it includes checked baggage, flexible changes, lounge access, or trip protection from the card used to book.

When showing sample deals in the MVP, TripSignal must label them as sample inventory. In production, prices should come from a provider such as Amadeus or Duffel with a search timestamp, cabin, baggage caveat, and bookability status.
