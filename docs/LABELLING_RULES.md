# Labelling rules

Written by Jasper, copied from the group proposal. Everyone follows these before annotating.

## Classes

| Id | Class |
|---|---|
| 0 | `fresh_apple` |
| 1 | `fresh_banana` |
| 2 | `fresh_orange` |
| 3 | `rotten_apple` |
| 4 | `rotten_banana` |
| 5 | `rotten_orange` |

The proposal's rule sheet spells these `rotton_*`. Keep the order above in Roboflow and the ids will match either way.

## Fresh or rotten

| Fruit | Fresh | Rotten |
|---|---|---|
| Apple | Smooth skin with normal colour; normal speckles | Any mold; bruises (brown patches); holes; broken or torn skin; wrinkled, shrivelled skin |
| Banana | Green to yellow peel; small brown spots on a yellow peel are still fresh | Peel mostly brown or black; large dark patches; mold; splits or holes; mushy or collapsing |
| Orange | Firm, even colours; a green bit near the peel is fine | Mold; white, green or blue fuzz; any bruises or sunken spots; mushy patches; holes; shrivelled skin |

## Boxes

- One box per fruit. Never put a box around a whole group.
- The box touches the fruit's edges, with no extra background.
- **Banana bunches:** box each banana seen on its own. Skip it if you cannot see where it starts and ends.
- **Partly hidden fruit:** box only the part you can see. Skip it if too little is visible, or if you cannot tell which fruit it is or whether it is rotten.

> **Open item for Jasper:** the proposal says "less than (idk a third or half) is visible". Pick one threshold before
> annotation starts, otherwise four people will apply four different rules.

## Do not label

- Peeled or cut fruit
- Other fruit
- Anything you cannot identify
