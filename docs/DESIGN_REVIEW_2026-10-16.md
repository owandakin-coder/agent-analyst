# ATZMA workspace redesign — 2026-10-16

## Research and design decisions

Reviewed published product patterns and primary design guidance. This was not
an inspection of private accounts or a claim that every current app screen was
reproduced. The goal is a coherent ATZMA workspace with fewer interruptions.

| Reference | Relevant pattern | ATZMA implementation |
|---|---|---|
| [Revolut 10](https://www.revolut.com/blog/post/revolut-10/) | Clear navigation and direct access to core destinations | Overview, Positions, Analytics, Orders and Settings, with three home shortcuts |
| [Robinhood portfolio navigation](https://robinhood.com/us/en/support/general-questions/check-your-portfolio/) | Distinct portfolio, charts, lists and activity areas | Separate holdings, analytics and order destinations; evidence has its own section within activity |
| [Apple navigation design](https://developer.apple.com/videos/play/wwdc2022/10001/) and [modality guidance](https://developer.apple.com/design/human-interface-guidelines/modality) | Hierarchical navigation; reserve modal interruption for a discrete task | Replace ten independent overlays with one detail surface and a visible Back control; sign-in remains the single modal transaction |
| [Material adaptive layout](https://m3.material.io/foundations/layout/canonical-examples/overview) | Adapt navigation and content to available space | Bottom navigation on smaller screens; sidebar and multi-column content on desktop |
| [W3C modal dialog pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/) | Focus containment, Escape and restoration of focus | Auth focus trap, inert background, Escape dismissal, explicit close control and focus restoration |

## Changes

- A wider desktop workspace, restrained dark palette, consistent borders,
  typography, spacing, line icons and touch targets. Responsive layouts replace
  the previous fixed 430px application width.
- One active detail destination. Profile, connection, model information, risk
  preferences and other utilities share the same content area; opening another
  destination cannot create a stack of overlapping windows.
- Browser Back/Forward participates in application navigation. Back restores
  the preceding destination and focus; hidden pages and panels are inert.
- A deliberate left swipe inside a mobile detail surface goes back. Vertical
  scrolling, inputs, buttons, charts, multi-touch, long drags, desktop input
  and the outer 28px reserved for browser/OS gestures do not trigger it.
- Existing form DOM is retained across history navigation. Broker refreshes
  preserve dirty fields. Async profile/connection responses cannot reopen a
  destination the user has left. Signing out prevents history from exposing
  personal panels again.
- Research evidence and execution queue use two section buttons instead of
  one long mixed page. Broker environment options and setup instructions are
  available through disclosures. System health is collapsed initially.
- Signed-out users can explore the locked interface without an automatic
  login popup. Private data still uses the existing authentication and broker
  verification gates. Unavailable balances use dashes instead of fictitious
  zero balances. The decorative synthetic sparkline is hidden.
- Accessible control names, visible keyboard focus, keyboard activation for
  legacy interactive rows, a skip link, reduced-motion support, and logical
  layout properties with RTL back-arrow handling.

## Verification

Browser QA covered the real local signed-out interface and an explicitly
synthetic, network-isolated fixture for account forms. No real credentials were
entered and no broker/control action was submitted. Verified:

- Compact viewport at 389 × 844 CSS pixels and desktop layout; no horizontal
  overflow in inspected views.
- One visible panel, Back/Forward, focus return, unsent field preservation,
  section switching, auth focus trapping, Escape and signed-out privacy.
- A synthetic TouchEvent integration path actually invoked the registered
  swipe handlers and navigated back. This is not physical-device Safari or
  Android testing; those browser/OS gesture interactions remain a practical
  validation limit.
- Browser console contained no runtime errors in the inspected fixture.
- Four Node tests cover route replacement/validation and gesture acceptance
  and rejection. Python checks require identical HTML/CSS/JS in the root and
  dashboard deployment copies. CI also runs the existing backend/API suite.

Local validation completed: 295 Python tests and four Node navigation tests
passed. JavaScript syntax, DOM IDs/structure, asset parity and workflow YAML
were checked. Desktop was inspected at 1280 × 853 CSS pixels.

The implementation uses the existing application and deployment pipeline.
Training, model weights, scheduler and broker execution logic are unchanged.
