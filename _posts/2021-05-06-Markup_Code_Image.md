---
classes: wide
title: "Markup: Post with Code/Images"
excerpt: "Displaying code and images in posts."
date: 2021-05-06
last_modified_at: false
header:
  teaser: "http://farm9.staticflickr.com/8426/7758832526_cc8f681e48_c.jpg"
---

## Code

If you want to have inline text use acutess `like this`.

For code block use three acutes followed by the language (html, python, etc):
```python
import pandas as pd
import numpy as np
```


## Figures (for images or video)

If you want to display two or three images next to each other responsively use `figure` with the appropriate `class`.  
Each instance of `figure` is auto-numbered and displayed in the caption.

#### One Figure

<figure>
	<a href="/assets/images/star_galaxy_1200x777.jpg"><img src="/assets/images/star_galaxy_1200x777.jpg"></a>
	<figcaption>Figure caption.</figcaption>
</figure>


#### Two Up

Apply the `half` class like so to display two images side by side that share the same caption.

```html
<figure class="half">
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/assets/images/image-filename-1.jpg"></a>
    <a href="/assets/images/image-filename-2-large.jpg"><img src="/assets/images/image-filename-2.jpg"></a>
    <figcaption>Caption describing these two images.</figcaption>
</figure>
```

<figure class="half">
	<a href="/assets/images/star_galaxy_1200x777.jpg"><img src="/assets/images/star_galaxy_1200x777.jpg"></a>
	<img src="/assets/images/star_galaxy_1200x777.jpg">
	<figcaption>Caption for both images: left has link, right does not.</figcaption>
</figure>

#### Three Up

Apply the `third` class like so to display three images side by side that share the same caption.

```html
<figure class="third">
	<img src="/images/image-filename-1.jpg">
	<img src="/images/image-filename-2.jpg">
	<img src="/images/image-filename-3.jpg">
	<figcaption>Caption describing these three images.</figcaption>
</figure>
```

<figure class="third">
	<img src="/assets/images/star_galaxy_1200x777.jpg">
	<img src="/assets/images/star_galaxy_1200x777.jpg">
	<img src="/assets/images/star_galaxy_1200x777.jpg">
	<figcaption>Three images.</figcaption>
</figure>

